(* Mathematica Package *)
(* Created by the Wolfram Language Plugin for IntelliJ, see http://wlplugin.halirutan.de/ *)

(* :Title: WolfDetectorUtils *)
(* :Context: WolfDetectorUtils` *)
(* :Author: Alec *)
(* :Date: 2021-06-23 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: 12.3 *)
(* :Copyright: (c) 2021 Alec *)
(* :Keywords: *)
(* :Discussion: *)

BeginPackage["WolfDetectorUtils`"];
(* Exported symbols added here with SymbolName::usage *)

Begin["`Private`"];

(* Reformats our YOLO network output into "boxes", "confidences", and "classes" *)
netBreakoutFn = Function[{anchors, grid, conv}, Block[
  (* We assume the net has been transposed to dimensions n x Anchors x dim x dim.  *)
  {
    boxes = conv[[1 ;; 4]],
    classes = conv[[6 ;;]],
    confidences = LogisticSigmoid[conv[[5]]]
  },
  (* We first need to construct our boxes to find the best fits. *)

  Block[{
    boxesScaled =
        Join[(( 2.0 / 13.0 * Tanh[boxes[[1 ;; 2]]] //
            TransposeLayer[{2 <-> 3, 3 <-> 4}]) + (grid //
            TransposeLayer[{1 <-> 3, 3 <-> 2}]) //
            TransposeLayer[{4 <-> 2, 3 <-> 4}]),
          (Transpose @ (4 * LogisticSigmoid[ boxes[[2 ;; 3]]]) *
              anchors // Transpose)]},
    <|
      "boxes" -> boxesScaled,
      "confidences" -> confidences,
      "classes" -> classes
    |>
  ]]];

breakoutLayer = FunctionLayer[Apply[netBreakoutFn]];

(*outputGraph[inputSize_, nClasses_, nAnchor_, gridLayer_, anchorLayer_] := NetGraph[<|*)
(*  "grid" -> gridLayer,*)
(*  "anchors" -> anchorLayer,*)
(*  "reshape" ->*)
(*      NetChain[{ReshapeLayer[{5, Automatic, inputSize/32,*)
(*        inputSize/32}], TransposeLayer[1 <-> 2]}],*)
(*  "breakout" -> breakoutLayer,*)
(*  "classSoftmax" -> SoftmaxLayer[1]*)
(*|>,*)
(*  {*)
(*    {"anchors", "grid", NetPort["Input"] -> "reshape"} -> "breakout",*)
(*    NetPort["breakout", "classes"] ->*)
(*        "classSoftmax" ->  NetPort["classes"]*)
(*  },*)
(*  "Input" -> {(5 + nClasses)*nAnchor, inputSize/32, inputSize/32}*)
(*];*)

(* Smooth L1 Loss Function, see https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html *)
sl1Loss = FunctionLayer[Apply[
  With[{diff = Abs[#boxes - #labels]},
    If[diff < 1.0,
      0.5 * diff^2,
      diff - 0.5]] &]];

(* Generalized Intersection Over Union (https://giou.stanford.edu) *)
GIOUGraph = NetGraph[
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

(* Returns a graph that computes our YOLO loss function for training. *)
lossGraph[nClasses_, nAnchor_, inputSize_, gridLayer_, anchorLayer_] := NetGraph[
  <|
    "grid" -> gridLayer,
    "anchors" -> anchorLayer,
    "reshape" ->
        NetChain[{ReshapeLayer[{5, Automatic, inputSize / 32,
          inputSize / 32}], TransposeLayer[1 <-> 2]}],
    "breakout" -> breakoutLayer,
    "targetBoxes" -> PartLayer[{All, 1 ;; 4}],
    "targetClasses" -> PartLayer[{All, 5 ;;}],
    "iou" -> NetMapThreadOperator[GIOUGraph, <|"label" -> 1|>] (* wow *),
    "maxes" -> AggregationLayer[Max, 2 ;;],
    "assignedMask" ->
        FunctionLayer[
          If[#iou > 0.5 || Abs[#iou - #max] < 0.002 , 1., 0.] &],
    "anyAssignedMask" -> AggregationLayer[Max, 1],
    "iouThreshold" ->
    (* This is needed because Network crashes without thresholding applied. *)
        FunctionLayer[
          If[#iou > 0.5  , If[#iou > 0.75, 1, 0.5 ],
            If[#iou > 0.25, .25, 0.0]] &],
    "maxIouThreshold" -> AggregationLayer[Max, 1],
    "confidenceLossExpanded" ->
        FunctionLayer[
          Apply[(#confidence - Ramp[#iou])^2 * (0.1 + 2 * #anyAssigned) &]],
    "LossConfidence" -> SummationLayer[],
    "boxLossUnmasked" -> NetMapThreadOperator[FunctionLayer[
      AggregationLayer[Total, 1][sl1Loss[#predictedBoxes, #targetBoxes]]&
    ], <|"targetBoxes" -> 1|>],
    "boxLossExpanded" -> FunctionLayer[#loss * #mask * 2 &],
    "LossBox" -> SummationLayer[],
    "classSoftmax" -> SoftmaxLayer[1],
    "classLossUnmasked" ->
        (* This is CrossEntropy Loss. The simple method does not work:
    NetMapThreadOperator[CrossEntropyLossLayer["Probabilities"], <|
    "Target"\[Rule] 1|>]*)
        NetMapThreadOperator[
          FunctionLayer[-Total[Log[#Input] * #Target] &], <|"Target" -> 1|>],
    "classLossMasked" -> FunctionLayer[#loss * Ramp[#iou ] * 2 &],
    "LossClass" -> SummationLayer[]
  |>,
  {
    (* Compute best matching IOUs *)
    NetPort["Target"] -> {"targetBoxes", "targetClasses"},
    {"anchors", "grid", NetPort["Input"] -> "reshape"} -> "breakout",
    {NetPort["breakout", "boxes"], "targetBoxes" } -> "iou" -> "maxes" ,
    {"iou", "maxes"} -> "assignedMask",
    (* Calculate loss for confidence *)
    "assignedMask" -> "anyAssignedMask",
    "iou" -> "iouThreshold" -> "maxIouThreshold" ,
    {NetPort["breakout", "confidences"], (*"maxiou"*)(*"anyAssignedMask"*)
      "maxIouThreshold", "anyAssignedMask"} ->
        "confidenceLossExpanded" ->
            "LossConfidence" -> NetPort["LossConfidence"],
    (* Calculate loss for box predictions *)
    {NetPort["breakout", "boxes"], "targetBoxes"} -> "boxLossUnmasked",
    {"boxLossUnmasked", "assignedMask" } ->
        "boxLossExpanded" -> "LossBox" -> NetPort["LossBox"],
    (*Calculate loss for class predictions *)
    NetPort["breakout", "classes"] -> "classSoftmax",
    {"classSoftmax", "targetClasses"} -> "classLossUnmasked",
    {"classLossUnmasked", "assignedMask"} ->
        "classLossMasked" -> "LossClass" -> NetPort["LossClass"]
    (*{"LossConfidence", "LossBox", "LossClass"} \[Rule] "Output"*)
  },
  "Target" -> {"Varying", 4 + nClasses},
  "Input" -> {(5 + nClasses) * nAnchor, inputSize / 32, inputSize / 32}
];

End[]; (* `Private` *)

EndPackage[]