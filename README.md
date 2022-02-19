# WR-AConvNet-AIHUB

본 저장소는 AIHUB 노이즈 제거 데이터의 응용 알고리즘 예시로 "[AConvNet](https://ieeexplore.ieee.org/document/7460942)" 기반의 날씨 인식 방법을 제안gksek..  
노이즈 제거 데이터는 [AI Hub](http://www.aihub.or.kr/) 에서 제공되며, 모델은 [AConvNet-PyTorch[1]](https://github.com/jangsoopark/AConvNet-pytorch) 를 기반으로 구축 되었다. 


## Introduction

구축 된 노이즈 제거 데이터는 화질 열화에 영향을 준 다양한 외부 요인을 포함한다. 
본 저장소에서 제안하는 날씨 인식 방법은 외부 요인 중 기상 현상과 관련 된 데이터를 이용하여 구축되었다.  
본 저장소에서 인식하고자 하는 기상 현상은 아래와 같다.
![figure001](https://user-images.githubusercontent.com/3586713/154791428-d2aeafce-f6b6-48ce-a809-03a90d091975.png)
- 보통/맑음(Normal)
- 미세먼지(Dust)
- 안개(Fog)
- 비(Rain)
- 눈(Snow)


![figure002](https://user-images.githubusercontent.com/3586713/154793554-633ec338-04dd-4f94-a186-8cd3f4714776.png)


## Installation

### Pre-requisites

- [Python 3.x](https://www.python.org/downloads/)

### Dependencies
- requirements.txt
```pipreqs
absl-py==1.0.0
numpy==1.21.5
onnxruntime==1.10.0
opencv_python==4.5.5.62
```


```shell
pip install -r requirements.txt
```

### Execution
```shell
python src/inference.py \ 
  --model_path=assets/model/weather-recognition.onnx \
  --image_path=samples/dust.jpg
  
# EX) Expected Output 
# ... inference_onnx.py:71] START
# ... inference_onnx.py:65] Prediction: dust
# ... inference_onnx.py:66] Elapsed Time: 0.007000923156738281
# ... inference_onnx.py:67] FPS: 142.83830540798257
# ... inference_onnx.py:75] FINISH
```


## Contacts
jspark@ivstech.co.kr

## License
MIT License

### Citation
```
@ARTICLE{7460942,
  author={S. {Chen} and H. {Wang} and F. {Xu} and Y. {Jin}},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Target Classification Using the Deep Convolutional Networks for SAR Images}, 
  year={2016},
  volume={54},
  number={8},
  pages={4806-4817},
  doi={10.1109/TGRS.2016.2551720}
}
```

### References
[1] Jangsoo Park (2022), AConvNet-PyTorch [Source Code]. https://github.com/jangsoopark/AConvNet-pytorch.  