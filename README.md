## Wine API

Wine api devoloped on flask with yolo to detect labels and extract text via ocr.

## Basics


Folders `static/`, `templates` are dependents of file `app_web.py`

`app_web.py` is only for web ui debugging (These three files can be deleted without affecting main app)


`test.py` is test the api

### Parameters

Identifiable parameters by the yolo model 

`AlcoholPercentage`, 
`AppellationQualityLevel`, 
`AppellationRegion`, 
`Country`, 
`DistinctLogo`, 
`EstablishedYear`, 
`MakerName`, 
`VintageYear`, 
`WineType`, 
`Sweetness`, 
`Organic`, 
`Sustainable`

## Installation

> [!NOTE]
> - `python3 -m venv venv` to create python virtual env
> - `./venv/scripts/activate` to activate venv in windows or `source/bin/activate` in linux
> - `run pip install -r req.txt` to install necessary packages
> - To run the program `python app.py`


### Info

> [!NOTE]
> - yolo v8s model is used for labelling and bounding box
> - Microsoft TrOCR is used for OCR Text extraction

> [!IMPORTANT]
> - use ultralytics==8.0.196 
> - numpy<2
