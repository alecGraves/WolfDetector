(* Mathematica Package *)
(* Created by the Wolfram Language Plugin for IntelliJ, see http://wlplugin.halirutan.de/ *)

(* :Title: WolfDetectorData *)
(* :Context: WolfDetectorData` *)
(* :Author: Alec *)
(* :Date: 2021-06-23 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: 12.1 *)
(* :Copyright: (c) 2021 Alec *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["WolfDetectorData`"];
(* Exported symbols added here with SymbolName::usage *)

Begin["`Private`"];


Clear[fromRectangle];
fromRectangle[rect_] := {rect[[1]], rect[[2]]};


Clear[scaleRectangles];
scaleRectangles[rects_, img_] := (fromRectangle /@ rects) //
    Transpose[#, {3, 2, 1}] & //
    # * ImageDimensions[img] & //
    Transpose[#, {3, 2, 1}] & //
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


ClearAll[xywhToRects];
xywhToRects[xywh_] := Block[{xs, ys, ws, hs},
  {xs, ys, ws, hs} = Transpose[xywh];
  Rectangle[{#[[1]], #[[2]]}, {#[[3]], #[[4]]}] & /@ (FunctionLayer[
    Clip[#in, {0, 1}] &][{xs - ws/2, ys - hs/2, xs + ws/2, ys + hs/2}] // Transpose )
];


ClearAll[getClassExtractor];
getClassExtractor[
  dataset_] := (Values /@ Values[dataset]) // Flatten //
    DeleteDuplicates // Sort //
    FeatureExtraction[#, "IndicatorVector",
      FeatureTypes -> "Nominal"] &;


Clear[trainingSample];
trainingSample[datum_, classExtractor_] := With[{
  img = Keys[datum],
  labels = Values[datum]
},
  With[{
    rectangles = Keys[labels],
    classes = Values[labels]
  },
    {img, Join[rectsToXYWH[rectangles], classExtractor[classes], 2] }
  ]];


Clear[ComputeBestAnchors];
ComputeBestAnchors[dataset_, nAnchor_] :=
    With[{allXYWH =
        rectsToXYWH[dataset // Values // Keys // Flatten[#, 1] &]},
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
                Join[means,
                  Table[
                    Extract[ means,
                      First @ Position[clusterLens, Max@clusterLens, {1}, 1] ],
                    nMissing]
                ]
              ],
              means
            ]]]]];


End[]; (* `Private` *)

EndPackage[]