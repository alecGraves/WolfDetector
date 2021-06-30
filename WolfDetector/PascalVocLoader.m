(* Mathematica Package *)
(* Created by the Wolfram Language Plugin for IntelliJ, see http://wlplugin.halirutan.de/ *)

(* :Title: PascalVocLoader *)
(* :Context: PascalVocLoader` *)
(* :Author: Alec *)
(* :Date: 2021-06-28 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: 12.1 *)
(* :Copyright: (c) 2021 Alec *)
(* :Keywords: parser to convert Pascal-Voc-format (https://gist.github.com/Prasad9/30900b0ef1375cc7385f4d85135fdb44)
dataset to WolfDetector dataset *)
(* :Discussion: *)

BeginPackage["PascalVocLoader`"];
(* Exported symbols added here with SymbolName::usage *)

LoadVoc::usage = "LoadVoc[path_] loads a pascal-voc style dataset from the specified `path`";

Begin["`Private`"];

(* Return list of XMLElements matching the key that you request in an XMLObject or XMLElement *)
Clear[findElements];
findElements[data_, key_String] := Cases[data, XMLElement[key, _, _], Infinity];

(* Convert list of {XMLElement[key1, _, value1], ...} to {value1, ...}  *)
Clear[extractValues];
extractValues[xmlElements_] := Map[
(*  Convert single-value list to unpacked value *)
  If[Length[#] == 1, First[#], #]&,
  ReplaceAll[xmlElements, XMLElement[_, _, value_] -> value]
];

(* Extract list of numeric, string, etc. data from keys within an XMLObject *)
Clear[fromXML];
fromXML[xml_, key_] := findElements[xml, key] // extractValues;

(* convert XMLObject from Voc annotation to WolfDetector box format *)
Clear[extractBoxesVoc];
extractBoxesVoc[xml_] := With[{
  filename = First[fromXML[xml, "filename"]],
  width = Interpreter["Number"][First[fromXML[xml, "width"]]] /. _Failure -> 0,
  height = Interpreter["Number"][First[fromXML[xml, "height"]]] /. _Failure -> 0,
  xmin = Map[Interpreter["Number"], fromXML[xml, "xmin"]],
  ymin = Map[Interpreter["Number"], fromXML[xml, "ymin"]],
  xmax = Map[Interpreter["Number"], fromXML[xml, "xmax"]],
  ymax = Map[Interpreter["Number"], fromXML[xml, "ymax"]],
  classnames = fromXML[xml, "name"]
},
  (*    Check size makes sense  *)
  If[ width == 0 || height == 0,
    (*      return empty list*)
    filename -> Missing["InvalidAnnotation"],
    (*      else return: image filename -> (WolfDetector box -> class) *)
    filename -> Map[
      ( Rectangle[#[[1;;2]], #[[3;;4]]] -> #[[5]] & ),
      Transpose[{N[xmin]/width, N[ymin]/height, N[xmax]/width, N[ymax]/height, classnames}]
    ]
  ]
];

Clear[extractBoxesVocFile];
extractBoxesVocFile[xmlPath_] := If[FileExistsQ[xmlPath],
  With[{data=Import[xmlPath, "XML",	"AllowRemoteDTDAccess"->False, "ReadDTD" -> False]}, extractBoxesVoc[data]],
  Missing["NotAvailable"]
];

Needs["GeneralUtilities`"];
Clear[LoadVoc];
LoadVoc[path_] := With[{
  xmlFiles = FileNames["*.xml", path, Infinity, IgnoreCase->True],
  imgFiles = FileNames[{"*.jpg", "*.jpeg", "*.png", "*.tiff"}, path, Infinity, IgnoreCase->True]
},
  With[{imageNameToBoxes = Apply[Association, ParallelMap[extractBoxesVocFile, xmlFiles]]},
    Map[# -> imageNameToBoxes[[ FileNameTake[#] ]]&, imgFiles] // DeleteCases[_->_Missing]
  ]
];

End[]; (* `Private` *)

EndPackage[]
