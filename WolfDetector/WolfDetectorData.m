(* Mathematica Package *)
(* Created by the Wolfram Language Plugin for IntelliJ, see http://wlplugin.halirutan.de/ *)

(* :Title: WolfDetectorData *)
(* :Context: WolfDetectorData` *)
(* :Author: Alec *)
(* :Date: 2021-06-23 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: 12.3 *)
(* :Copyright: (c) 2021 Alec Graves *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["WolfDetectorData`"];
(* Exported symbols added here with SymbolName::usage *)

GetClassExtractor::usage = "GetClassExtractor[dataset] gets a one-hot encoding function for your dataset.";

ComputeBestAnchors::usage = "ComputeBestAnchors[dataset, n]: Computes optimal `n` bounding box anchors from a dataset";

ToTrainingDataset::usage = "ToTrainingDataset[dataset, classExtractor] converts a dataset of rectangles \
to a format that can be used for training.";

DrawBoxesLabel::usage = "DrawBoxesLabel[img_, labels_] Draws label boxes onto an img given the image";

ToLabel::usage = "ToLabel[outboxes_, outclasses_, classExtractor_] Takes network outputs outboxes and outclasses \
and returns labeled boxes in the form {Rect[...] -> 'class', ...}";


Begin["`Private`"];

(* convert Rect[{xmin, ymin}, {xmax, ymax}] to {{xmin, ymin}, {xmax, ymax}} *)
Clear[fromRectangle];
fromRectangle[rect_] := {rect[[1]], rect[[2]]};

(* Scale WolfDetector format Rectangle boxes to image boxes. *)
Clear[scaleRectangles];
scaleRectangles[rects_, img_] := (fromRectangle /@ rects) //
    Transpose[#, {3, 2, 1}] & //
(*        Transpose to 2*2*n, where first dim is x, y, second is min, max, third is number of boxes   *)
(*        Invert y-coordinate for y-down image frame.    *)
    {#[[1]], 1-#[[2]]} & //
(*        Scale the images   *)
    # * ImageDimensions[img] & //
    Transpose[#, {3, 2, 1}] & //
(*        Convert back to rectangles   *)
    Apply[Rectangle, #] & /@ # &;


Clear[rectsToXYWH];
rectsToXYWH[rectangles_] := With[{
  reshaped = (fromRectangle /@ rectangles)  //
      Transpose[#, {3, 2, 1 }] &
},
  With[{
    centers = (Mean /@ reshaped) // Transpose ,
    sizes = ((Subtract[ #[[2]], #[[1]] ] &) /@ reshaped) // Transpose
  },
    Join[centers, sizes, 2]
  ]];


Clear[xywhToRects];
xywhToRects[xywh_] := Block[{xs, ys, ws, hs},
  {xs, ys, ws, hs} = Transpose[xywh];
  Rectangle[{#[[1]], #[[2]]}, {#[[3]], #[[4]]}] & /@ (
    FunctionLayer[Clip[#in, {0, 1}] &][{xs - ws/2, ys - hs/2, xs + ws/2, ys + hs/2}] // Transpose
  )
];


Clear[ToLabel];
ToLabel[outboxes_, outclasses_, classExtractor_] := (#[[1]] -> #[[2]]) & /@
    Transpose[{xywhToRects[outboxes], NetDecoder[classExtractor][outclasses]}];


Clear[DrawBoxesLabel];
DrawBoxesLabel[img_, labels_] := HighlightImage[
  img,
  Labeled[ #[[1]], #[[2]] ] & /@ Transpose[{scaleRectangles[Keys[labels], img], Values[labels]}]
];

DrawBoxesLabel[imgFilename_String, labels_] := With[{img=Import[imgFilename]}, DrawBoxesLabel[img, labels]];

Clear[GetClassExtractor];
GetClassExtractor[dataset_] := (Values /@ Values[dataset]) // Flatten //
    DeleteDuplicates // Sort //
    NetEncoder[{"Class", #, "UnitVector"}] &;


Clear[trainingSample];
trainingSample[datum_, classExtractor_] := With[{
  img = Keys[datum] /. x_String -> File[x],
  labels = Values[datum]
},
  With[{
    rectangles = Keys[labels],
    classes = Values[labels]
  },
    img -> Join[rectsToXYWH[rectangles], classExtractor[classes], 2]
  ]];


Clear[ToTrainingDataset];
ToTrainingDataset[dataset_, classExtractor_] := Map[trainingSample[#, classExtractor]&, dataset];


Clear[ComputeBestAnchors];
ComputeBestAnchors[dataset_, nAnchor_] :=
    With[{allXYWH = rectsToXYWH[dataset // Values // Keys // Flatten[#, 1] &]},
      (* get the WH components*)
      With[{allWH = allXYWH[[All, 3 ;; 4]] },
        With[{clustered = FindClusters[allWH, nAnchor, Method -> "KMeans"]},
          With[{
            means = Mean /@ clustered (*This is hacky, it would be nice to direcly extract from clustering process. *)
          },
            (*We could just return means, but FindClusters may generate fewer anchor box clusters than we want.
            To handle this, we we will check if enough means were found and repeat the most \
                common mean until we have the desired number of anchors. *)
            If[Length[clustered]  < nAnchor,
              With[{
                clusterLens = Length /@ clustered,
                nMissing = nAnchor - Length[clustered]
              },
                Join[
                  means,
                  Table[
                    Extract[ means, First @ Position[clusterLens, Max@clusterLens, {1}, 1] ],
                    nMissing
                  ]
                ]
              ],
              means
            ]]]]];


End[]; (* `Private` *)

EndPackage[]
