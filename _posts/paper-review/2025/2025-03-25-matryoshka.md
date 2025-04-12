---
title: "[논문 리뷰] Matryoshka Representation Learning"
author: invhun
date: 2025-03-25 15:00:00 +0900
categories: [Paper Review, Representation Learning]
tags: [Matryoshka, Representation Learning, Adaptive Retrieval, Reranking]
math: true
pin: false
---

> Matryoshka Representation Learning   
> Neurips 2022   
> **Aditya Kusupati**, **Gantavya Bhatt**, Aniket Rege, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, Ali Farhadi   
> University of Washington, Google Research, Harvard University   
> [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/c32319f4868da7613d78af9993100e42-Paper-Conference.pdf)], [[ArXiv](https://arxiv.org/pdf/2205.13147)]


해당 리뷰는 Representation Learning에 대한 기초지식이 있다는 가정 하에 진행하였음

# 1. Abstract & Introduction

### 기존 연구 문제점
- Representation은 ML 시스템의 핵심 요소로 여러 다운스트림 작업에 사용되나, 고정된 capativty(dimension)를 가지는 representation은 필요한 작업에 비해 과도하거나 부족할 수 있음
    - 계산된 representation을 응용에 사용할 때, 계산 비용은 임베딩 차원, 데이터 크기, 레이블 공간에 따라 증가하며, 웹 규모에서는 이 비용은 representation을 계산하는 비용을 초과함
- 기존 연구는 유연성을 얻기 위하여, 여러 저차원 모델을 학습하거나, 다양한 capacity의 하위 네트워크를 공동 최적화하거나, 사후 압축을 수행하였음, 하지만 이는 학습/유지관리에서의 오버헤드, 많은 비용이 드는 forward pass, 정확도 하락 등의 문제가 발생함

### 제안 방법
![fig1](https://1drv.ms/i/c/af5642aec05791fb/IQQjMXrhrkcQTp5wZg01zNj4AS7et4XKBe2izD4XzmjBVjM?width=1024)
- 제안 방식 Matryoshka Representation Learning (MRL)은 기존 파이프라인을 최소한으로 수정하여 추가 비용 없이 다양한 세분화 수준에서 정보를 인코딩하고, 유연한 representation을 생성할 수 있음
- MRL을 사용하여 최대 14배 빠르면서 정확한 대규모 Classification, Retrieval을 수행함
- 다양한 모달리티: Vision(ResNet, ViT), Vision+Language(ALIGN), Language(BERT), 그리고 웹 규모 데이터(ImageNet-1K/4K, JFT-300M, ALIGN Data)에 대한 원할한 적용을 보여줌


# 2. Related Work

### Efficient Classification and Retrieval
- 추론 중 분류 및 검색의 효율성은 레이블 수 (L), 데이터 크기 (N), Representation 크기 (d)에 대한 선형 의존성이 있으며, 이는 RAM, 디스크 및 프로세서에 동시에 부담을 줌
- 레이블 수에 대한 의존성은 근사 최근 이웃 검색 (ANNS)나 기본 계층 구조를 활용한 연구가 있음
- Representation 크기의 경우, 차원 축소, 해싱 기법 드잉 있으나, 상당한 정확도의 감소를 초래함.
- 실제 검색 시스템은 대규모 임베딩 기반 검색에 의해 구동되며, 증가하는 웹 데이터에 따라 비용이 증가, HNSW 기법이 이를 해결하기 위해 쓰이지만, 정확도는 유지되나 RAM과 디스크에 대한 오버헤드 비용이 발생함
- MRL은 다양한 용량의 신경망을 패킹하는 연구와 차별점이 있으며, 다중 충실도의 표현을 학습하여 선형 복잡도 의존성 해결

# 3. Matryoshka Representation Learning

![fo1](https://1drv.ms/i/c/af5642aec05791fb/IQRBR3PVbeRATpcrGGyuNyW0AVBzcNk2s5ngGAWhkOvn41U?width=1024)
- 주어진 라벨이 있는 데이터셋 $$D = {(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)}$$에서 각 데이터 포인트 $$x_i$$와 그에 대응하는 레이블 $$y_i$$ 에 대해 다중 클래스 분류 손실을 최적화
- 중첩 차원 집합 $$M$$은 $$m \in M$$ 형태로 각 차원이 선택되며 표현크기가 낮은 정보 병목에 도달할 떄 까지 절반씩 나누며 선택함, 여기서는 {8, 16, ..., 1024, 2048}을 선택. 또한 각 중첩 차원에 대해 별도의 선형 분류기 $$W(m)$$ 가 사용됨 
- 손실 함수는 다중 클래스 소프트맥스 교차 엔트로피 손실 함수로 정의되며, 선형 분류기와 데이터 포인트의 표현 $$F(x_i; \theta_F)_{1:m}$$을 입력으로 받습니다. 즉 앞에서부터 m개의 차원을 선택하였다고 볼 수 있음
- 여기서 $$c_m$$은 각 중첩 차원의 상대적 중요성을 나타내며, 여기서 모든 m에 대해 1로 설정하였음
- 즉 이 방식은 MRL이 고차원 표현 d에 대해 O(log(d))의 중첩 차원만 최적화함으로써, 선택된 차원 사이의 값들에 대한 보간 표현을 생성한다고 볼 수 있음
- MRL_E는 MRL의 변형으로 모든 선형 분류기 간에 가중치를 묶어서 메모리 비용을 절감하는 방식. 즉 $$W(m) = W_{1:m}$$으로 정의하여 공통 가중치를 사용$$

- 상세 알고리즘 코드는 다음과 같다
![algo1](https://1drv.ms/i/c/af5642aec05791fb/IQQRCBKkSudlSb0d5dMOpNDYAfDTyiNwzUIYl8uYLt69EDU?width=1024)
![algo2](https://1drv.ms/i/c/af5642aec05791fb/IQQaGf46UGYERoyDf0xXeIoYAaIkyvVQZw02H1B0JBQp5b8?width=1024)



# 4. Application

## 4.1. Representation Learning
- MRL을 여러 Representation Learning에 적용
    - Vision(Supervised learning): ResNet50 on ImageNet-1K, ViT-B/16 on JFT-300M
    - Vision+Language(Contrastive learning): ViT-B/16 & BERT on ALIGN
    - Language(Masked modelling): BERT on English Wikipeida & BooksCorpus
- 모든 실험에 하이퍼파라미터는 독립적으로 학습된 baseline model과 동일한 하이퍼파라미터를 사용, 최적의 하이퍼파라미터를 탐색하진 않았음
- ResNet은 {8, 16, ... 1024, 2048} 차원을, ViT-B/16은 {12, 24, ..., 384, 768} 차원을 사용
- 이를 독립적으로 학습된 저차원 표현(FF), 차원 축소(SVD), subnet방법(Slim. Net), 가장 높은 Capacity를 가지는 FF모델에서 무작위 선택된 특성(Rand)와 비교를 수행함

- 여기서 MRL은 하나의 ResNet에 Algorithm2를 적용해서 학습한다면, FF는 각 ResNet의 FN만 "torch.nn.Linear(k, num_classes)"으로 바꾸었다고 보면 된다.


## 4.2. Classification
![fig2](https://1drv.ms/i/c/af5642aec05791fb/IQRdkmwAAtEXSoG71RjvFDtfAdNxPB5c0r4N-Bff_HpGkZ8?width=1024)
- ImageNet-1K에서 학습 및 선형 분류 평가를 수행하였을 때, 모델의 각 표현은 FF가 유사하거나 약간 더 높음.

![fig3](https://1drv.ms/i/c/af5642aec05791fb/IQTsvOueanUQTLwA8UyjyZp3AZUaCtb2HcR7v11pOKRuRdc?width=1024)
![t2](https://1drv.ms/i/c/af5642aec05791fb/IQTD_f1redOzQr6RHL0h-FIxAe6cyZQ9R8u0GE61ymxRFyY?width=1246&height=482)
- 1 Nearest Neighbor(1-NN) 평가를 하였을 때, 낮은 차원에서 2% 이상의 높은 정확도를, 모든 차원에서 동등한 정확도를 유지함.

![fig4](https://1drv.ms/i/c/af5642aec05791fb/IQSCTVAUuChvT5eHjKp4iFxHAWl4D4w8nTxGYDQgCP7Lhpc?width=770&height=782)
- JFT-300M, ALIGN으로 학습 후 ImageNet-1K에서 1-NN 평가를 수행, 이때 높은 계산 비용(대규모 데이터)으로 인해, Image-Net 1k로 학습때와 달리 Rand방식만 비교에 사용.
- 더 높거나 유사한 정확도를 달성

![fig5](https://1drv.ms/i/c/af5642aec05791fb/IQTfw0TKctDWTrTTJcft4eDNAdOHFdCi0Pdai_6bdabOUSw?width=790&height=788)
- MRL은 중첩된 차원에서 학습된 정보가 전체 차원에 걸쳐 interpolation(보간) 되기 때문에, 중간 차원에 대한 정보로 자연스럽게 확산됨.
- 즉 O(log(d))의 중첩된 차원만 학습하여, 전체 차원 (d)에 비례하는 자원과 시간을 절약하면서도, 중간 차원에 대한 성능 저하 없이 높은 정확도를 유지할 수 있음

### 4.2.1. Adaptive Classification
![fig6](https://1drv.ms/i/c/af5642aec05791fb/IQSMpLiUaCgjT7c-TFENkJNoAYe-er7TAe3U54JVwm9Yk1k?width=802&height=780)
![t3](https://1drv.ms/i/c/af5642aec05791fb/IQQTOoox77swRoR1RS_tsVg0AVxPN3qghBvoYY9lnw8bkHk?width=1612&height=504)
- 전통적인 model cascade 방식과 달리 MRL은 여러번의 forward pass를 필요로 하지 않음
- 임계값 학습: ImageNet-1K의 1만개 Valid set으로 각 중첩 차원에 대해 최대 softmax 확률의 임계값을 학습, 이 임계값을 사용하여 MRL 모델이 더 높은 차원 표현(8->16->32, ...)으로 전환할 시점을 결정
- 100개의 샘플에 대해 grid search를 수행, 각 임계값에 대해 분류 정확도를 계산하고, 가장 높은 정확도를 제공하는 가장 작은 임계값을 설정.
- 이 과정을 반복하여 각 차원에 대한 임계값을 얻음
- 추론 과정에선 나머지 40,000개의 샘플을 사용, 가장 작은 차원 8에서 시작하여, 예측 신뢰도와 최적 임계값을 비교함. 신뢰도가 임계값보다 작을 경우 차원을 증가시킴.
- 이로 인해 37 representation 차원만으로 76.3%의 분류 정확도를 달성, FF-512 대비 14배 더 작은 차원 크기
- representation 크기의 누적합에 대한 가중 평균을 계산하여도, 여전히 8.2배 효율적

## 4.3. Retrieval
![fig7](https://1drv.ms/i/c/af5642aec05791fb/IQR05dSTK5pMQo6L2RuFL6EoAakZNaUjox1oiA98D4D4xqo?width=790&height=780)
- image retrieval 성능을 mAP@k로 평가를 수행
- MRL은 종종 가장 정확하며, FF보다 최대 3% 더 높은 성능을 보임
- MRL은 다양한 세분화에서 정확한 검색을 수행할 수 있으며, 웹 규모 데이터베이스에 대해 여러 forward 패스의 추가 비용이 발생하지 않음
    - forward pass: 4GFLOPs
    - retrieval: 2.6GFLOPs(ImageNet-1K: 1.3M), 8.6GFLOPs(ImageNet-4K:4.2M)
- FF모델은 차원별 독립적인 데이터베이스를 생성하며, 이는 저장 및 전환 비용이 매우 큼
- 실제 어플리케이션에서는 정확한 검색(O(dN))대신 HNSW와 같은 Aproximate Nearest Neighbor Search (ANNS)를 사용하여 (O(d \log(N)))으로 대체함, 이는 추가 메모리 오버헤드가 발생하지만, 정확도의 최소한의 감소를 가져옴
- MRL은 이러한 ANNS 파이프라인의 일부로 사용되는 모든 벡터 압축 기술과 상호보완적이며, 효율성과 정확도 간의 트레이드 오프를 향상시킬 수 있음

### 4.3.1. Adaptive Retrieval
![fig8](https://1drv.ms/i/c/af5642aec05791fb/IQQ_vJsncgaXTqOt40YgnFjHAakuFC3bDbjuaETz81ybOho?width=1024)
- AR은 주어진 쿼리에 대해 데이터베이스에서 낮은 차원의 표현(e.g. $$D_s=16$$)을 사용하여 이미지 후복 목록 (K=200)을 생성 한 후, 더 높은 차원의 표현(e.g. $$D_r=48$$)을 사용하여 reranking 하는 방식으로 진행.
- fig8은 MRL을 사용한 AR과 단일 검색 간의 tradeoff를 보여줌, AR은 모든 경우에 단일 검색의 최적선 위에 위치험
- AR은 단일 검색에 비해 동일한 정확도를 가지면서 이론적으로 약 128배, 실제로 14배 빠름(HNSW사용시)

#### Funner Retrieval
- $$D_s$$와 $$D_r$$ 선택을 쉽게 하기 위해, 일관된 AR을 위한 Funner retrieval을 제안.
- 차원을 증가시키며, 이전 후보목록을 반복적으로 reranking하는 방식
- e.g. 후보 목록: [200->100->50->25->10], 차원 크기: [16->32->64->128->256->2048]

# 5. Further Analysis and Ablations

### Robustness
![t17](https://1drv.ms/i/c/af5642aec05791fb/IQRI76KRyfbbQKzXX6ry8dPSAaUaHdIS7YnfUZRhXuyHz3Q?width=1626&height=512)
- ImageNet-1K에서 학습 후, ImageNetV2/R/A/Sketch에서의 FF와 비교 평가
- 대부분의 경우 FF보다 우수한 성능을 보이며, 강건성을 보여줌

### Few-shot and Long-tail Learning
![t15](https://1drv.ms/i/c/af5642aec05791fb/IQQ3jUu8cKqCSZNbs3m28ObwAb6HmMciZfq8j-VFGAlfhaQ?width=1624&height=1202)
- Few-shot Learning: Few shot과 클래스 수에 대해 FF 표현과 유사한 성능을 보입

![t16](https://1drv.ms/i/c/af5642aec05791fb/IQQy9DCIDMrEQ6lWw3CH-TT3AZcAVhWAJ5IVvlFPtrZUJSE?width=1632&height=1470)
- FLUID 프레임워크에서 MRL이 새로운 클래스에서 최대 2% 높은 정확도를 제공하지만, 다른 클래스의 정확도는 희생하지 않는 것을 관찰

### Disagreement across Dimensions
![fig12](https://1drv.ms/i/c/af5642aec05791fb/IQRkYOqu91HDQ5FJB8GyW9SvARXxt3bzyTE3SCgCK6mmJqs?width=1024)
- MRL의 정보 패킹(information packing)이 차원 capacity 증가에 따라 점진적으로 증가하는 경향을 보이나, 특정 인스턴스와 클래스에서는 낮은 차원에서 더 높은 정확도를 보이는 경우가 있음
- 즉 적절한 차원으로 라우팅 할 경우, 최대 4.6%의 분류 정확도를 향상시킬 수 있음

### Superclass Accuracy
![fig10](https://1drv.ms/i/c/af5642aec05791fb/IQTGMG8Cph-uTKN5AAPqy8NwAZbI4NGvQj6LZeO1XCemSE0?width=804&height=692)
- 그림 3에서 볼 수 있듯이, overall 정확도는 fine-grained class에서 차원이 감소될 때 급격히 감소함을 확인할 수 있음.
- superclass 즉 coarse-grained class에서 보았을 때는, 이러한 경향이 완화됨을 보이며, MRL이 모든 차원에서 FF보다 더 높은 정확도를 보임을 확인할 수 있음

## 5.1. Ablation
- 여기의 내용은 모두 간략하게 핵심 문장을 나열하였고, 표와 상세 내용은 모두 appendix에 있다고 기술되어 있음
- 저자는 fine-tuning에 적용 가능함을 보여줌, 또한 로스의 최적 가중치를 사용하여, 저차원 표현의 정확도를 손실 없이 향상시킬 수 있음을 보여줌
- 또한 차원 선택의 경우, 초기엔 매우 낮은 차원을 피하는 것이 좋으며, 현재의 로그 간격 spacing이 효과적임을 보여줌
- 검색 성능이 데이터셋의 복잡성에 따라 후보 목록 및 특정 차원 길이 이후에 포화됨을 보여주며, 효율적인 검색을 위해 최적의 후보 목록 크기와 차원을 설정해야함을 시사함

# Appendix K: Ablation Studies

## K.1. MRL Training Paradigm

### Matryoshka Representations via Finetuning
![t26](https://1drv.ms/i/c/af5642aec05791fb/IQRjd9lnCqANT4SdCSK3t5uHAQ76P2uLz4mKrCV5N24Un4c?width=1620&height=680)
- nesting이 명시적으로 학습되지 않은 모델에 적용이 될 수 있는지를 확인하기 위해, 사전학습된 FF-2048-ResNet50모델에 MRL레이어를 추가하였음
- ResNet50 아키텍처의 층을 다양한 경우로 unfreeze하여 nesting을 강제하는데 필요한 비선형성의 정도를 관찰
- MRL 선형 레이어만 fine-tuning하는 것으론 부족하며, conv+ReLU를 추가할수록 d=8에서 정확도가 5%=>60%로 향상됨(10epoch), 이는 40epoch동안 RML을 처음부터 학습한 경우와 비교하였을 때 6% 차이에 불가함
- 차원이 증가함에 따라 이 차이는 점점 감소하여 d=64 이상에서는 1.5% 이내로 줄어들었음

### Relative Importance
![t27](https://1drv.ms/i/c/af5642aec05791fb/IQQK1egjTZbCSLjigN9xNGVxAVvrWufsGdWhhTiKxGw40sA?width=1616&height=682)
- 저차원에서 성능을 향상시키기 위해, 학습 로스의 상대적 중요성을 조정하였음
- MRL-8boost의 경우 8차원을 2로, 나머지 차원은 그대로 1의 중요성으로 두었음
- 이때, 8차원에서 top-1정확도가 3%, 16~256에서 약간의 향상, 그리고 512~2048에서는 최대0.1%의 성능 하락이 관찰되었음
- 상대적 중요도에 대해 강건성을 보임과 동시에, 특정 차원에 대해 최적의 정확도를 위해 조정될 수 있음을 나타냄

### Matryoshka Representations at Arbitrary Granularities
![t29](https://1drv.ms/i/c/af5642aec05791fb/IQRIVdw5yuBOR7m_O6fCfYNmAbKMLj9KiE98oswZQzODov4?width=804&height=766)
- MRL-Uniform: {8, 212, 416, 620, 824, 1028, 1232, 1436, 1640, 1844, 2048}
- MRL-Log: {8, 16, 32, 64, 128, 256, 512, 1024, 2048}
- Uniform 방식과 비교하였을 때, log방식이 비용 측면에서 우수하며, 낮은 차원에서 성능이 좋았음, 고 차원에서는 포화상태에 이름 -> 전반적으로 Log방식이 우수

### Lower Dimensionality
![t28](https://1drv.ms/i/c/af5642aec05791fb/IQRUJggoRy1TRZO8K2pJFhJWAU08gHmyalh0I4OjrZ1DSQU?width=812&height=854)
- 8차원보다 작은 차원에 대해서 테스트를 수행
- 8차원 미만의 정확도는 낮아서 배포에 적합하지 않으며, 학습 난이도가 증가하였음
- 또한 고차원에서 전체적인 작은 정확도 감소를 초래하였으며, 이는 최적화가 더 어려운 작은 차원이 포함된 것 때문으로 판단됨

# Review
- 아이디어는 비교적 간단하다고 할 수 있지만, 그렇다고 쉽게 생각해내기는 어려운 아이디어라고 생각되며, 정말 다양한 extensive한 실험을 수행하였음.
- 연구의 필요성을 논리적으로 수치적으로 잘 나타냈으며, 기존 연구와의 비교를 잘 수행하였음.
- 연구적으로 다양한 분야에 적용될 가능성이 높으며, application 측면에서도 우수한 연구라고 생각이 됨.
- 저자는 이미지 도메인 뿐만 아니라, 자연어 처리 분야, 멀티모달 분야에도 적용하였으나, 하지만 대부분의 실험이 이미지 도메인에 치우쳐있으며, 자연어와 멀티모달에 대해서는 깊이 탐구하였다고 보기는 어려움. 후속 연구가 많이 진행되었으나, 멀티모달 분야의 임베딩 압축 면에서는 현재 탐구가 아직 부족한 상황이라고 생각됨

