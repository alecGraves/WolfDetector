(* Mathematica Package *)
(* Created by the Wolfram Language Plugin for IntelliJ, see http://wlplugin.halirutan.de/ *)

(* :Title: WolfDetector *)
(* :Context: WolfDetector` *)
(* :Author: Alec Graves *)
(* :Date: 2021-06-22 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: 12.3 *)
(* :Copyright: (c) 2021 Alec *)
(* :Keywords: *)
(* :Discussion: *)

(*

  ::Usage::

WolfDetector is an object detection library for #WolfLang (Wolfram Language).



  :: 1. Dataset ::

To use it, you must first have the dataset you want to train on:

```
myDataset = {
<img> -> { Rectangle[{x_min, y_min}, {x_max, y_max}] -> "Class1", ...},
  ...
};
```

where:
x_min is the smallest x value for a bounding box
y_min is the smallest y value for a bounding box
x_max is the largest x value for a bounding box
y_max is the largest y value for a bounding box

Note: x_min and x_max are encoded as fractions of the image width (0.0 to 1.0, where 0.0 is the left),
and y_min and y_max are encoded as fractions of the image height (0.0 to 1.0, where 0.0 is the top).


To train, we first have to get critical information about our data (the number of classes and anchors to use).
Run the following code to do this step.


    dataInfo = WolfDetectorDataInfo[dataset]


    Out[]= <|"classExtractor" -> FeatureExtractorFunction[...], "anchors"->{...}|>



  :: 2. Model Setup ::

You must set up your custom object detection model and its output layer to use for your problem.
To do this, you can set the input image size (default 416x416)


Run one of the following to build the neural network:


    {net, outputLayer} = BuildWolfDetector[dataInfo]

    (* set input image size to (224, 224): *)
    {net, outputLayer} = BuildWolfDetector[dataInfo, 224]



  :: 3. Model Training ::


First, we must pre-process our dataset into a trainable format

    trainingDateset = WolfDetectorTrainingDataset[myDataset];

Next, we must create our loss function:

    loss = WolfDetectorLoss[outputLayer]


Finally, we can train our network:

    trained = NetTrain[
      net,
      trainingDataset,
      LossFunction -> loss,
      ValidationSet->Scaled[0.1],
      MaxTrainingRounds->20
    ]



  :: 4. Model Inference ::

We first want to combine our output layer to our trained model:

    predictor = NetChain[{trained, outputLayer}]

Now we have trained our model, we want to run it on a new image.
To do this, we first evaluate our model to get the output:

    pred = predictor[myNewImg];

Next, we postprocess our prediction into useful output data:

    outputData = WolfDetectorOutput[prediction, dataInfo]

Note we can also specify threshold parameters to our output non-max-suppression algorithm:

    minObjectConfidence = 0.5;
    maxBoxOverlap = 0.1;
    outputData = WolfDetectorOutput[prediction, dataInfo, minObjectConfidence, maxBoxOverlap]

Finally, we can visualize this output data:

    WolfDetectorVisualize[outputData, dataInfo]

*)

BeginPackage["WolfDetector`"];
(* Exported symbols added here with SymbolName::usage *)

WolfDetectorDataInfo::usage = "WolfDetectorDataInfo[dataset_, nAnchors_:5] computes and returns information \
about the data that is used to make predictions.";

BuildWolfDetector::usage = "BuildWolfDetector[dataInfo_, imageSize_:416] returns a neural network graph \
for object detection.";

WolfDetectorTrainingDataset::usage = "WolfDetectorTrainingDataset[dataset_, dataInfo_] converts a correctly formatted \
dataset into the format used by the WolfDetectorLossFunction.";

WolfDetectorLoss::usage = "WolfDetectorLoss[outputLayer] builds and returns the loss function \
for training the WolfDetector object detection network.";

WolfDetectorOutput::usage = "WolfDetectorOutput[prediction_, dataInfo_, minObjectConfidence_, maxBoxOverlap_] performs
filtering to remove overlapping and low-confidence predictions.";

WolfDetectorVisualize::usage = "WolfDetectorVisualize[img_, labels_] returns a graphic with labels drawn on img";

WolfDetectorLoadDataset::usage = "WolfDetectorLoadDataset[path_, format_] loads a dataset at `path` \
with a specified `format`. ex: WolfDetectorLoadDataset[\"C:\\my_datasets\\dataset1\", \"voc\"]";

WolfDetectorLoadDataset::fmtsupport = "No loader found for dataset format: `1`. \
\"voc\" format is supported.";


Begin["`Private`"];

(* Import our utility packages*)
Get["WolfDetectorData`"];
Get["WolfDetectorNetYolo`"];
Get["PascalVocLoader`"];

(* Define our functions *)

Clear[WolfDetectorLoadDataset];
WolfDetectorLoadDataset[path_, format_] := With[{
(*  Choose the loader to use based on the passed in string code. *)
  loader = <|
(*    Add new dataset loaders here. *)
    "voc" -> LoadVoc
  |>[ format ]
},
  If[MissingQ[loader],
    (Message[WolfDetectorLoadDataset::fmtsupport, format]; $Failed),
(*    else*)
    loader[path]
  ]
];


Clear[WolfDetectorDataInfo];
WolfDetectorDataInfo[dataset_, nAnchors_:5] := <|
  "classExtractor" -> GetClassExtractor[dataset],
  "anchors" -> ComputeBestAnchors[dataset, nAnchors]
|>;


Clear[BuildWolfDetector];
BuildWolfDetector[dataInfo_, imageSize_:416] := With[{
    anchors = dataInfo["anchors"],
    nAnchors = dataInfo["anchors"] // Length,
    nClasses = dataInfo["classExtractor"][["Labels"]] // Length
  },
  With[{
    yolo = BuildYolo[imageSize, nClasses, nAnchors],
    yoloOut = BuildYoloOutput[imageSize, nClasses, anchors]
  },
    {yolo, yoloOut}
  ]];


Clear[WolfDetectorTrainingDataset];
WolfDetectorTrainingDataset[dataset_, dataInfo_] := ToTrainingDataset[dataset, dataInfo["classExtractor"]];


Clear[WolfDetectorLoss];
WolfDetectorLoss[outputLayer_] := BuildYoloLoss[outputLayer];


Clear[WolfDetectorOutput];
WolfDetectorOutput[prediction_, dataInfo_, minObjectConfidence_:0.5, maxBoxOverlap_:0.1] := Block[{outBoxes, outClasses},
  {outBoxes, outClasses} = NonMaxSuppression[prediction, minObjectConfidence, maxBoxOverlap];
  If[Length[outClasses] > 0, ToLabel[ outBoxes, outClasses, dataInfo[["classExtractor"]] ], {}]
];


Clear[WolfDetectorVisualize];
WolfDetectorVisualize[img_, labels_] := If[Length[labels] > 0, DrawBoxesLabel[img, labels], img];

End[]; (* `Private` *)

EndPackage[]