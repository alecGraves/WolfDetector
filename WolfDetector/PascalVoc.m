(* Mathematica Package *)
(* Created by the Wolfram Language Plugin for IntelliJ, see http://wlplugin.halirutan.de/ *)

(* :Title: PascalVoc *)
(* :Context: PascalVoc` *)
(* :Author: Alec *)
(* :Date: 2021-06-28 *)

(* :Package Version: 0.1 *)
(* :Mathematica Version: 12.1 *)
(* :Copyright: (c) 2021 Alec *)
(* :Keywords: parser to convert Pascal-Voc-format (https://gist.github.com/Prasad9/30900b0ef1375cc7385f4d85135fdb44)
dataset to WolfDetector dataset *)
(* :Discussion: *)

BeginPackage["PascalVoc`"];
(* Exported symbols added here with SymbolName::usage *)

ConvertVoc::usage = "";

Begin["`Private`"];

Clear[xmlToAssoc];
Needs["GeneralUtilities`"]; (* for ToAssociations - which now can be replaced with ResourceFunction["ToAssociations"] *)
xmlToAssoc[xml_XMLObject, lvl_:Infinity] := With[{xmlElems=Cases[xml, XMLElement[_, _, __]]},
  xmlElems // Replace[#, {XMLElement[x_, _, y__] -> (x -> y)}, lvl]& // GeneralUtilities`ToAssociations
];


extractBoxes[xmlPath_] := With[{data=Import[xmlPath, "XML",	"AllowRemoteDTDAccess"->False, "ReadDTD" -> False]},
  With[{assoc = xmlToAssoc[data]},
    blah
  ]
];

ConvertVoc[path_] := With[{
  xmls = FileNames["*.xml", path, Infinity, IgnoreCase->True],
  imgs = FileNames[{"*.jpg", "*.jpeg", "*.png", "*.tiff"}, path, Infinity, IgnoreCase->True]
},

];

End[]; (* `Private` *)

EndPackage[]