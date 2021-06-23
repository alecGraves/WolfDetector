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



  :: 2. Model Setup ::

You must set up your custom object detection model to use for your problem.
To do this, you can set the following parameters (or use default values):

1. An input image size (must be evenly divisible by 32, default 416x416)
2. Number of box size anchors to use (default 5).
3. Number of possible output classes (you can extract this from your dataset).

Run one of the following to complete setup:


    net = BuildWolfDetector[myDataset]

    (* set input image size (416, 416): *)
    net = BuildWolfDetector[myDataset, {416, 416}]

    (* set number of anchor boxes 5: *)
    net = BuildWolfDetector[myDataset, {416, 416}, 5]

    (* set number of classes (2), must be >= number of classes in your dataset. *)
    net = BuildWolfDetector[myDataset, {416, 416}, 5, 2]



  :: 3. Model Training ::

First, we must create our loss function:


    loss = WolfDetectorLoss[]


Next, we must pre-process our dataset into a trainable format

    trainingDateset = WolfDetectorTrainingDataset[myDataset];

Finally, we can train our network:

    trained = NetTrain[net, trainingDataset, LossFunction -> lossGraph, ValidationSet->Scaled[0.1], MaxTrainingRounds->20]



  :: 4. Model Inference ::

Now we have trained our model, we want to run it on a new image.
To do this, we first evaluate our model to get the output:

    prediction = trained[myNewImg];

Next, we postprocess our prediction into useful output data:

    outputData = WolfDetectorOutput[prediction]

Note, we can also specify threshold parameters to our output Non-max-suppression algorithm:

    minObjectConfidence = 0.5;
    maxBoxOverlap = 0.1;
    outputData = WolfDetectorOutput[prediction, minObjectConfidence, maxBoxOverlap]

Finally, we can visualize this output data:

    WolfDetectorVisualize[outputData]

*)

BeginPackage["WolfDetector`"];
(* Exported symbols added here with SymbolName::usage *)

Begin["`Private`"];



End[]; (* `Private` *)

EndPackage[]