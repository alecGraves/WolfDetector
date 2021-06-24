(* Mathematica Package *)
(* Created by the Wolfram Language Plugin for IntelliJ, see http://wlplugin.halirutan.de/ *)

(* :Title: WolfDetectorNetYolo *)
(* :Context: WolfDetectorNetYolo` *)
(* :Author: Alec *)
(* :Date: 2021-06-23 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: 12.3 *)
(* :Copyright: (c) 2021 Alec *)
(* :Keywords: *)
(* :Discussion: *)

(* YOLOv2 loss function in Mathematica *)

BeginPackage["WolfDetectorNetYolo`"];
(* Exported symbols added here with SymbolName::usage *)

BuildYolo::usage = "BuildYolo[inputSize_, nClass_, nAnchor_] Returns pre-trained YOLO for the specified
square input image size and a new head for the specified number of classes and anchors. \
Image size must be divisible by 32, and the size / 32 must be odd (e.g. 416/32 = 13, 224/13 = 7)";

BuildYoloOutput::usage = "BuildYoloOutput[inputSize_, nClasses_, nAnchor_, anchors_] Returns layer \
converting yolo convolutional output to 'boxes', 'confidences' and 'classes'";

BuildYoloLoss::usage = "BuildYoloLoss[outputLayer_] Returns YOLOv2 loss function NetGraph \
(assumes YOLO output is in the format of GetOutputGraph";

NonMaxSuppression::Usage = "Filters YOLO output, removing low-confidence and overlapping boxes using thresholds";


Begin["`Private`"];

Clear[makeInput];
makeInput[inputSize_Integer] :=
    NetEncoder[{"Image", inputSize}] /;
        Divisible[inputSize, 32] && OddQ[inputSize/32];


Clear[makeReshape];
(*  note this seems to confine us to square images... *)
makeReshape[inputSize_] := NetChain[
  {
    ReshapeLayer[{Automatic, 2, inputSize[[2]]/16, 2}],
    TransposeLayer[{1 <-> 3, 1 <-> 4, 1 <-> 2}],
    ReshapeLayer[{Automatic, inputSize[[2]]/32, inputSize[[2]]/32 }]
  }
];


Clear[resizeConv];
(* This is horribly hacky; surely there is some way around needing \
this *)
resizeConv[convLayer_, newSize_] := (convLayer
    // NetReplacePart[#, "Input" -> NetEncoder[{"Function", Identity, newSize}]] &
    // NetReplacePart[#, "Input" -> None] &);


Clear[toplessYOLO];
(*
Make a yolo, minus the final conv layer, with some square size input, \
defaulting to 3x416x416.

Note the size must be divisble by 32 and the input/32 must be odd \
(the output should be odd for optimal performance)
*)
toplessYOLO[inputSize_Integer : 416] /;
    Divisible[inputSize, 32] && OddQ[inputSize/32] := With[{
  conv =
      NetExtract[NetModel["YOLO V2 Trained on MS-COCO Data"], "Conv"]
},
  With[{
    convBase =
        NetTake[conv, {"conv32", "conv512"},
          "Input" -> makeInput[inputSize]]
  },
    NetGraph[<|"conv" ->  convBase,
      "final1" ->
          NetTake[conv, {"conv1024", "conv_final_1"},
            "Input" -> convBase[["Output"]]],
      "final2" ->
          NetChain[{resizeConv[NetExtract[conv, "conv_final_2"],
            convBase[["Output"]]]}],
      "reshape" ->
          NetChain[{makeReshape[convBase[["Input", "Output"]]]}],
      "combine" -> CatenateLayer[]
    |>, {"conv" -> {"final1", "final2" -> "reshape"} -> "combine"
    }] // NetFlatten[#, 1] &
  ]];


Clear[newOutputConv];
newOutputConv[nClass_, nAnchor_] :=
    ConvolutionLayer[(nClass + 5)*nAnchor, 1];

Clear[newYoloTop];
newYoloTop[nClass_, nAnchor_] := NetGraph[{
  (* Maybe consider using the pre-
  trained version of this first layer? *)
  NetChain[{
    ConvolutionLayer[1024, 3, PaddingSize -> {{1, 1}, {1, 1}}],
    BatchNormalizationLayer["Epsilon" -> 1*10^-6],
    ParametricRampLayer["Slope" -> 0.1]
  }],
  (* This is our output layer. *)
  newOutputConv[nClass, nAnchor]
}, {1 -> 2}];


Clear[BuildYolo];
BuildYolo[inputSize_, nClass_, nAnchor_] := NetChain[{
  toplessYOLO[inputSize],
  newYoloTop[nClass, nAnchor]
}] // NetFlatten[#, 1] & // NetInitialize;



Clear[BuildYoloOutput];
(* Converts Yolo conv output to usable array data *)
BuildYoloOutput[inputSize_, nClasses_, anchors_] := NetGraph[<|
  "anchors" -> NetArrayLayer["Array" -> anchors],
  "grid" -> NetArrayLayer["Array" ->
      (Table[{x, y}, {x, 1, inputSize/32}, {y, 1, inputSize/32}] - 1) / (inputSize/32.0) ],
  "reshape" ->
      NetChain[{ReshapeLayer[{5, Automatic, inputSize/32, inputSize/32}], TransposeLayer[1 <-> 2]}],
  "breakout" -> FunctionLayer[Apply[Function[{anchorsIn, grid, conv}, Block[
    (* This is a really bad function because in 12.2, FunctionLayer was not compiling nice functions. *)
    (* ... *)
    (* We assume the net has been reshaped to dimensions n x Anchors x dim x dim.  *)
    {
      boxes = conv[[1 ;; 4]],
      classPredictions = conv[[6 ;;]],
      confidences = LogisticSigmoid[conv[[5]]]
    },
    (* We first need to construct our boxes to find the best fits. *)

    Block[{
      boxesScaled =
          Join[((1.0/(inputSize/32.0)*Tanh[boxes[[1 ;; 2]]] // TransposeLayer[{2 <-> 3, 3 <-> 4}]) +
              (grid // TransposeLayer[{1 <-> 3, 3 <-> 2}]) // TransposeLayer[{4 <-> 2, 3 <-> 4}]),
            (Transpose @ (3*LogisticSigmoid[ boxes[[2 ;; 3]]]) * anchorsIn // Transpose)]},
      <|
        "boxes" -> boxesScaled,
        "confidences" -> confidences,
        "classes" -> classPredictions
      |>
    ]]]]],
  "classSoftmax" -> SoftmaxLayer[1]
|>,
  {
    {"anchors", "grid", NetPort["Input"] -> "reshape"} -> "breakout",
    NetPort["breakout", "classes"] -> "classSoftmax" ->  NetPort["classes"]
  },
  "Input" -> {(5 + nClasses)*Length[anchors], inputSize/32, inputSize/32}
];


Clear[sl1Loss];
(* Smooth L1 Loss Function, see https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html *)
sl1Loss[] := FunctionLayer[Apply[
  With[{diff = Abs[#boxes - #labels]},
    If[diff < 1.0,
      0.5 * diff^2,
      diff - 0.5]] &]];


Clear[GIOUGraph];
(* Generalized Intersection Over Union (https://giou.stanford.edu) *)
GIOUGraph[] := NetGraph[
  <|
    (* width/height *)
    "awh" -> PartLayer[{3 ;; 4}],
    "bwh" -> PartLayer[{3 ;; 4}],
    (* Centers *)
    "axyc" -> PartLayer[{1 ;; 2}],
    "bxyc" -> PartLayer[{1 ;; 2}],
    (*min/max x y values*)
    "axymax" -> FunctionLayer[Apply[#1 + #2 / 2 &]] ,
    "axymin" -> FunctionLayer[Apply[#1 - #2 / 2 &]] ,
    "bxymax" -> FunctionLayer[Apply[#1 + #2 / 2 &]] ,
    "bxymin" -> FunctionLayer[Apply[#1 - #2 / 2 &]] ,
    (* iou calculation *)
    "nwh" ->
        FunctionLayer[
          Apply[ Ramp[
            ThreadingLayer[Min, 2][#axymax, #bxymax] -
                ThreadingLayer[Max, 2][ #axymin, #bxymin]] &]],
    "n" -> AggregationLayer[Times, 1],
    "nc" -> FunctionLayer[
      Apply[ThreadingLayer[Max, 2][ #axymax, #bxymax] -
          ThreadingLayer[Min, 2][#axymin, #bxymin] &]],
    "ac" -> AggregationLayer[Times, 1],
    "u" ->
        FunctionLayer[
          Apply[AggregationLayer[Times, 1][#awh] +
              AggregationLayer[Times, 1][#bwh] - #n &]],
    "iou" -> FunctionLayer[Apply[#n / #u - (#ac - #u) / #ac &]]
  |>,
  {NetPort["boxes"] -> {"awh", "axyc"},
    NetPort["label"] -> {"bwh", "bxyc"},
    {"axyc", "awh"} -> {"axymax", "axymin"},
    {"bxyc", "bwh"} -> {"bxymax", "bxymin"},
    {"axymax", "bxymax", "axymin", "bxymin"} -> {"nwh", "nc"},
    {"nwh"} -> "n",
    {"nc"} -> "ac",
    {"awh", "bwh", "n"} -> "u",
    {"n", "u", "ac"} -> "iou" -> NetPort["iou"]}
];


Clear[BuildYoloLoss];
(* Returns YOLO v2 - ish loss function *)
BuildYoloLoss[outputLayer_] := NetGraph[
  <|
    "breakout" -> outputLayer,
    "targetBoxes" -> PartLayer[{All, 1 ;; 4}],
    "targetClasses" -> PartLayer[{All, 5 ;;}],
    "iou" -> NetMapThreadOperator[GIOUGraph[], <|"label" -> 1|>] (* wow *),
    "maxes" -> AggregationLayer[Max, 2 ;;],
    "assignedMask" ->
        FunctionLayer[
          If[#iou > 0.5 || (Abs[#iou - #max] < 0.002) , 1., 0.] &],
    "anyAssignedMask" -> AggregationLayer[Max, 1],
    "iouThreshold" ->
(*     This is needed because NetTrain crashes without thresholding applied (in v12.3).*)
        FunctionLayer[Floor[#iou*64.0]/64.0 &],
    "maxIouThreshold" -> AggregationLayer[Max, 1],
    "confidenceLossExpanded" ->
        FunctionLayer[
          Apply[(#confidence - Ramp[#iou])^2 * (0.1 + 2 * #anyAssigned) &]],
    "LossConfidence" -> SummationLayer[],
    "boxLossUnmasked" -> NetMapThreadOperator[FunctionLayer[
      AggregationLayer[Total, 1][sl1Loss[][#predictedBoxes, #targetBoxes]]&
    ], <|"targetBoxes" -> 1|>],
    "boxLossExpanded" -> FunctionLayer[#loss * #mask * 2 &],
    "LossBox" -> SummationLayer[],
    "classLossUnmasked" ->
(*         This is CrossEntropy Loss. The simple method does not work:*)
(*    NetMapThreadOperator[CrossEntropyLossLayer["Probabilities"], <|"Target"\[Rule] 1|>]*)
        NetMapThreadOperator[
          FunctionLayer[-Total[Log[#Input] * #Target] &], <|"Target" -> 1|>],
    "classLossMasked" -> FunctionLayer[#loss * Ramp[#iou ] * 2 &],
    "LossClass" -> SummationLayer[],
    "Loss" -> TotalLayer[]
  |>,
  {
(*     Compute best matching IOUs*)
    NetPort["Input"] -> "breakout",
    NetPort["Target"] -> {"targetBoxes", "targetClasses"},
    {NetPort["breakout", "boxes"], "targetBoxes" } -> "iou" -> "maxes",
    {"iou", "maxes"} -> "assignedMask",
(*     Calculate loss for confidence*)
    "assignedMask" -> "anyAssignedMask",
    "iou" -> "iouThreshold" -> "maxIouThreshold" ,
    {NetPort["breakout", "confidences"],
      "maxIouThreshold", "anyAssignedMask"} ->
        "confidenceLossExpanded" ->
            "LossConfidence"(* -> NetPort["confidence"]*),
(*     Calculate loss for box predictions*)
    {NetPort["breakout", "boxes"], "targetBoxes"} -> "boxLossUnmasked",
    {"boxLossUnmasked", "assignedMask" } ->
        "boxLossExpanded" -> "LossBox"(* -> NetPort["box"]*),
(*    Calculate loss for class predictions*)
    {NetPort["breakout", "classes"], "targetClasses"} -> "classLossUnmasked",
    {"classLossUnmasked", "assignedMask"} ->
        "classLossMasked" -> "LossClass"(* -> NetPort["class"]*),
    {"LossConfidence", "LossBox", "LossClass"} -> "Loss"
  },
  "Target" -> {"Varying", 4 + First[NetExtract[outputLayer, "classes"]] (*nClasses*)},
  "Input" -> NetExtract[outputLayer, "Input"]
];

Clear[calculateOverlap];
calculateOverlap[box_, boxes_] :=  Block[{
  awh = box[[3 ;; 4]],
  bwh = boxes[[3 ;; 4]],
  axyc = box[[1 ;; 2]],
  bxyc = boxes[[1 ;; 2]],
  axyMax, axyMin, bxyMax, bxyMin, nwh, n, u, iou
},
  axyMax = axyc + awh/2;
  axyMin = axyc - awh /2;
  bxyMax = bxyc + bwh/2;
  bxyMin = bxyc - bwh/2;
  nwh = Ramp[ThreadingLayer[Min, -1][{axyMax, bxyMax}] - ThreadingLayer[Max, -1][{axyMin, bxyMin}]];
  n = AggregationLayer[Times, 1][nwh];
  u =  AggregationLayer[Times, 1][awh] + AggregationLayer[Times, 1][bwh] - n;
  iou = n / u
];

Clear[computeNonOverlapping];
computeNonOverlapping[inConfThreshold_, boxPreds_,
  classPredsInteger_, iouThreshold_ : 0.5, selectedIdxs_ : {}] :=
    If[Length[inConfThreshold] > 0,
      If[Length[inConfThreshold ] > 1,
        Block[{
          sameClass = Cases[Rest[inConfThreshold],
          _?(classPredsInteger[[#]] == classPredsInteger[[First[inConfThreshold]]] &)],
          overlaps,
          toRemove
        },
          toRemove = If[Length[sameClass] > 0,
            overlaps = calculateOverlap[ boxPreds[[First @ inConfThreshold]], Transpose @ boxPreds[[sameClass]] ];
            sameClass[[  Cases[Range[Length[sameClass]], _?(overlaps[[#]] > iouThreshold &)]  ]]
            ,
            {}
          ];
          computeNonOverlapping[
            DeleteCases[Rest[inConfThreshold], _?(MemberQ[toRemove, #] &)],
            boxPreds, classPredsInteger, iouThreshold,
            Append[selectedIdxs, First[inConfThreshold]]]
        ],
        Append[selectedIdxs, First[inConfThreshold]]
      ],
      selectedIdxs
    ];

Clear[NonMaxSuppression];
NonMaxSuppression[netOut_, confThreshold_ : 0.3, iouThreshold_ : 0.5, maxBoxes_ : 100] := Block[{
  boxPreds = netOut[["boxes"]] // Transpose[#, {4, 1, 2, 3}] & // Flatten[#, 2] &,
  classPreds = netOut[["classes"]] // Transpose[#, {4, 1, 2, 3}] & // Flatten[#, 2] &,
  confPreds = netOut[["confidences"]] // Flatten,
  classPredsInteger,
  sortedIdxs,
  inConfThreshold,
  nmsIdxs
},
  classPredsInteger = Map[First@Ordering[#, -1] &, classPreds];
  sortedIdxs = Reverse[ Ordering[confPreds, - Min[Abs[maxBoxes], Length[confPreds]]] ];
  inConfThreshold = Cases[sortedIdxs, _?(confPreds[[#]] >= confThreshold &)];
  nmsIdxs = computeNonOverlapping[inConfThreshold, boxPreds, classPredsInteger, iouThreshold];
  {boxPreds[[nmsIdxs]], classPreds[[nmsIdxs]]}
];

End[]; (* `Private` *)

EndPackage[]