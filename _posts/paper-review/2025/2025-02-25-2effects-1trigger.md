---
title: "[논문 리뷰] Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information Imbalance in Contrastive Vision-Language Models"
author: invhun
date: 2025-02-25 15:00:00 +0900
categories: [Paper Review, Multimodal Learning]
tags: [Contrastive Learning, Multimodal Learning, Modality Gap, Object Bias, CLIP]
math: true
pin: false
---

> Expertized Caption Auto-Enhancement for Video-Text Retrieval   
> arxiv, 5 Feb 2025   
> **Simon Schrodi**, **David T. Hoffmann**, Max Argus, Volker Fischer, Thomas Brox   
> University of Freiburg, Bosch Center for Artificial Intelligence   
> [[paper](https://openreview.net/pdf?id=uAFHCZRmXk)] [[github](https://github.com/lmb-freiburg/two-effects-one-trigger)]

# 1. Abstract & Introduction

![fig1](https://1drv.ms/i/c/af5642aec05791fb/IQQHZz4o7VaGRpSkGtL0rHzNAc6y_pE8dawuTtvExSZRFAo?width=340&height=310)

### 기존 연구 문제점
- 직관적으로 모달리티 갭이 성능을 제한할 것으로 예상되지만, 갭이 성능에 미치는 영향은 불확실하였으며, 분석이 충분히 이루어지지 않음
- 기존 연구는 VLM이 attribute(속성) 태스크에 비해 객체 태스크에서 성능이 현저히 낮은 것을 이유로 객체에 대한 편향을 제안하였음, 하지만 편향에 대한 평가는 제대로 정의되지 않았고 성능 저하로만 평가가 수행됨
- 모달리티 갭과 객체 편향이 발생하는 근본적인 원인에 대한 분석이 부족함

### 제안 방법(기여)
- 모달리티 갭의 구조와 특성을 심층적으로 분석하여, 성능 향상과의 관계를 규명함
- 모달리티 간 정보 불균형은 모달리티 갭과 객체 편향을 초래하며, 객체 편향은 객체 태스크와 부정적인 상관관계가 없음
- 모달리티 갭을 주도하는 임베딩 차원은 소수임
- 객체 편향을 측정하기 위해 Mathinc Object Attribute Distance (MOAD)를 제안함

# 2. Related work

- 모달리티 갭에 대한 기존 연구의 분석이 있으나, 아직 초기단계에 불과함
    1. 원뿔효과: 원뿔효과는 임베딩 공간이 좁은 원뿔 형태로 제한되는 현상으로, 서로 다른 모달리티가 무작위 초기화로 동일한 원뿔을 사용할 가능성이 낮기 때문에 모달리티 갭이 발생함
    2. 온도 매개변수의 영향을 분석
    3. 모달리티 갭이 이미지와 텍스트의 임베딩의 범위에 수직적임을 발견함
- 해당 논문에서는 일부 차원이 모달리티를 분리하는 주된 요인임을 발견하고, 정보 불균형이 모달리티 갭의 주요 원인임을 주장함

- VLM에 대해, 데이터의 중요성에 대한 연구, 일반화 및 강건성, 학습된 특징 분석, 조합 가능성 및 사회적 편향에 대한 연구가 활발히 진행되었고 최근에는 VLM이 객체에 편향되어 있다는 가설이 제안되었음
- 해당 논문에서는 VLM이 객체에 편향되어 있음을 검증하지만, 객체 편향이 적은 VLM이 반드시 더 나은 성능을 보이지는 않음을 발견하였음

# 3. Experimental Setup

- 실제 데이터와 완전히 제어 가능한 합성 데이터를 사용하여 실험을 수행
### Pretrained Constrastive VLMs
- CLIP 및 SigLIP과 같은 총 **98**개의 VLM을 분석에 사용, 중간 규모와 대규모 데이터셋에서 사전 훈련된 VLM을 구분하였음
- 심층 분석을 위해 CLIP ViT-B/16, SigLIP ViT-B/16을 사용
- 이 정도는 해야 분석 논문이 되는구나

### Evaluation Protocols
- 사전 학습된 VLM을 MSCOCO, ImageNet, MiT-States, UT-Zappos 데이터셋에서 표준 평가 프로토콜을 사용하여 평가
- 설명, 객체 클래스 또는 속성 앞에 "a photo of" 프롬포트 추가
- Retrieval에서 R@1, 제로샷 객체 및 속성 인식에선 Top-1 accuracy를 사용

### Fully-controlled, synthetic data.
![fig2](https://1drv.ms/i/c/af5642aec05791fb/IQQHRF0mIIcfTZMM_SThwe2lAQnW7diOvAiV-MqeT3nlPME?width=195&height=256)
- 모달리티 갭과 객체 편향에 대한 정보 불균형의 영향을 연구하기 위해 MorPho-MNIST를 기반으로 한 완전 제어 가능한 데이터셋인 Multimodal Attributes and Digits(MAD)를 구축
- 두께, 부풀림, 균열과 같은 변형 또는 왜곡 작업을 잠재적 요인(속성)으로 사용, 크기, 색상을 추가하였음
- 캡션의 경우 숫자 클래스와 다른 요인을 단어로 매핑, 무작위로 연결하여 생성, 정보 불균형 영향 분석을 위해 각 캡션 내 속성 수를 다양하게 조정 (e.g. "1-thickening-swelling-fractures-large-blue")

### Experiments on real data.
- 현실적인 설정에서 가설을 검증하기 위해 CLIP RN 50 모델을 CC12M에 대해 학습, 캡션 조작을 수행

# 4. Parting With False Intuitions About the Modality Gap

## 4.1. Does a Smaller Modality Gap Lead to Better Performance?

- "더 작은 모달리티 갭이 더 나은 성능으로 이어지는가?"에 대한 질문에 답하기 위해, CLIP 및 SigLIP을 비롯한 98개의 VLM에 대한 다운스트림 성능과 모달리티 갭을 평가하였음

- $$L2M := || \frac{1}{N} \sum_{i=1}^{N} x_i - \frac{1}{N} \sum_{i=1}^{N} y_i ||$$
- 기존 모달리티 갭 측정 지표인 L2M은 단순 평균 간 L2거리로, 이미지-텍스트 쌍의 매치를 고려하지 않으며, "does not take the effectively used space in to account", 즉 임베딩 공간에서 실제로 의미있는 정보를 나타내는 영역을 고려하지 않음.
- 이러한 한계로 해결하기 위해 Relatve Modality Gap (RMG)를 제안함
    ![fo1](https://1drv.ms/i/c/af5642aec05791fb/IQSPO5yanWC_QJNEaZG5dWFYAYZdeOxp-35fmXPHjmC6wio?width=858&height=147)
    - 분자는 일치하는 이미지-텍스트 쌍에 대한 갭을 측정
    - 분모는 모달리티 간 근사를 통해 효과적으로 사용된 공간을 고려함
    - 이때 거리함수는 코사인 비유사도(1 - 코사인 유사도)를 [0,1]로 스케일링 하여 사용.

![fig3](https://1drv.ms/i/c/af5642aec05791fb/IQSwUpNoDrQFQb7LdhrQJEZcAWQvFRcurXgmO1Dc990TZQ0?width=944&height=283)
![t1](https://1drv.ms/i/c/af5642aec05791fb/IQQyheTDlkDeRoUgEVdfv7IoAVASq5SPYVTdjIJWdMXJPu4?width=1024)
- 그림3과 표1은 더 큰 모달리티 갭이 역설적으로 더 나은 다운스트림 성능과 상관관계가 있음을 보여준다.
- 하지만 이것이 더 큰 모다리티 갭이 더 나은 성능으로 이어짐을 의미하지는 않으며, 표1에서 나타나듯 모델이나 임베딩 크기와 같은 다른 요인이 성능에 더 강한 영향을 미쳐, 모달리티 갭의 부정적인 영향이 가려지는 것으로 보인다.

![t3](https://1drv.ms/i/c/af5642aec05791fb/IQSngaIwdA9TRpVHd9Cvgj-aAXSY4d0_DHI9CJ1vjsxdAbY?width=1024)
- (appendix)
- 표3은 데이터셋을 통제했을 때, 70%(RAG), 60%(L2M)에서 모달리티 갭과 성능 간 부정적인 상관관계가 있음을 보여주며, 나머지는 통계적으로 유의미성을 발견하지 못하였음(p-value가 0.05이하일 때 통계적으로 유의미한 수치이며)
- 더 작은 갭이 더 나은 성능과 어느정도의 상관관계가 있음을 보여주며, 모달리티 갭은 해결한 가치가 있는 문제라 주장
- 하지만 LAION-400M을 제외하면 강한 부정적 상관관계를 보이지는 않는 듯 하며, 강한 긍정적 상관관계를 보이는 데이터셋도 있어서, 주장의 근거가 부족함(동일한 지적을 한 리뷰어도 있음)

## 4.2. Few Embedding Dimensions Drive the Modality Gap

### Do all embedding dimension contribute to the gap?
![fig4](https://1drv.ms/i/c/af5642aec05791fb/IQSwCwzOW-rDTIY1rHlYQcD3ARHv1U6WKAXV4hAJM2gYDbU?width=944&height=562)
- 그림 4a에서 대부분의 임베딩 차원이 매우 유사한 평균을 가지지만, 몇몇 차원에서는 뚜렷한 차이를 보이는 것을 확인
- 그림 4b에서 각 모달리티에서 가장 큰 평균을 가지는 차원을 플롯하였을 때, 이들만으로 모달리티를 완벽히 분리하는 것을 발견, 또한 이 차원들은 한 모달리티 내에서 큰 분산을 가지지만, 다른 모달리티에서는 미미한 분산을 보임
- (appendix) 리뷰어가 평균이 가장 높은 차원인지, 두 모달간 평균 차이가 가장 큰 차원인지를 질문하였고, 저자는 전자로 기술한 것이 맞고, 후자에 대해서도 추가적으로 실험을 하여 그림 12를 추가하였음

### Can we close the gap post-hoc?
- 모달리티 갭을 좌우하는 임베딩 차원을 사후에 제거하면 성능이 향상될 것 인가?
- 그림 4c는 모달리티 갭과 다운스트림 성능이 급격히 감소한 후, 성능이 부분적으로 회복됨을 보여줌
- 이러한 변화의 원인은 가장 큰 영향력을 가진 임베딩 차원의 제거 및 재정규화로 인해 발생하는 cross-model 이웃의 큰 변화를 설명함
- 이전 연구에서 테스트된 사후 접근 방식에서도 유사한 결과를 관찰할 수 있음
    1. 모달리티 갭 벡터를 계산하고, 이 벡터를 추가하여 갭을 줄이는 방식으로 일반적으로 성능 저하를 유발
    2. ideal word apporach: 간단한 이동이 갭을 줄이고, 평균 유사성 증가시키지만, 성능 향상으로 이어지지 않음
- 갭을 줄이는 조정이 효과적이기 위해서는 서로 다른 모달리티에서 유사한 이웃 관계를 가져야하며, 이웃 관계가 다르면, 모달리티를 줄이더라도, 정렬이 이루어지지 않기 때문
    (즉 이미지에서 특정 샘플의 가까운 이웃들이, 텍스트에서 동일한 샘플의 이웃과 유사하여야함)

# 5. Object Bias Is a Caption Presence Bias

![fig5](https://1drv.ms/i/c/af5642aec05791fb/IQSF7HSEkzBDR47VlXG4Nj0tAS8uIgcWyV4RNQ20KqpzWBk?width=936&height=435)
![fo2](https://1drv.ms/i/c/af5642aec05791fb/IQRSEzZ0VVQ7QLEF7UMEKjNrAcIwJ9p4r5LOYqYs-1-kC7o?width=824&height=78)
- Matching Object Attribute Distance (MOAD)를 제안
- 오브젝트의 positive pair 유사도와 negaative pair의 유사도 차이 평균과 같은 방식으로 계산한 attribute의 평균의 차이를 나타낸 것으로, 양의 값은 object로의 bias, 음의 값은 attribute로의 bias를 뜻함
- 그림 5(a)는 MODA값과 attribute 다운스트림 성능을 나타낸 것으로, 대규모 데이터셋으로 사전학습한 모델의 경우 object bias가 작은 것을 확인할 수 있음
- 또한 object로의 bias와 attribute 다운스트림 성능과의 유의미한 상관관계는 확인할 수 없음
- 그림 5(b)를 통해 object task과 attribute task의 성능에는 양의 상관관계(medium-to-strong correlations)가 있음을 확인할 수 있음. 중간규모 데이터셋으로 사전학습한 모델의 경우는 애매함

### Is object bias explained by the global word frequencies of the dataset?

![fig6](https://1drv.ms/i/c/af5642aec05791fb/IQTV5nJdkY5zRqZbDXZJWi3CATroZ_-X_ExfaKXDhk5hJ5k?width=475&height=600)
- 그림 6(a)는 LAION-2B의 캡션에 실제로 attribute과 객체보다 더 자주 언급되는것을 보여주며, 객체에 대한 편향이 학습데이터셋의 단어 빈도에서 기인하지는 않음을 확인할 수 있음.
- 저자는 객체에 대한 편향이 자연어에서 객체의 샘플별 "prevalence"에 있음을 가정함.
    즉, 사람들은 가장 눈에 띄는 객체와 그의 몇가지 속성만을 언급하는 경향이 있음
    따라서 이미지가 주어졌을 때, 단어의 조건부 확률이 $$p(word|image)$$ 편향을 유발한다고 가정함.
- 이를 검증하기 위해 MAD를 사용하여, 주요 요소가 항상 캡션에 포함되도록 하고 다른 요소 중 하나를 무작위로 샘플링하여 더섯개의 VLM을 학습시킴
- 그림 6(b)는 각 모델이 학습 중에 "prevalence"가 있었던 요소에 대한 편향이 있음을 보여줌.
- 또한 이미지 인코더는 가장 가능성이 높은 캡션과 맞춰야하기 떄문에, 이미지 인코더에서 편향이 더 크게 나타나는 것을 확인할 수 있음

# 6. Information Imbalance Triggers Modality Gap and Object Bias
## 6.1. How Does Information Imbalance Cause the Modality Gap and Object Bias?

### The origin of the object bias.
- 정보 불균형(이미지가 상대적으로 많은 정보)은 인코더들이 다른 모달리티의 정보를 알 수 없기 떄문에, 가장 가능성이 높은 요소(주로 객체)에 집중하고, 임베딩을 정렬하기 어렵게 만
- 이는 이전 섹션의 캡션 존재 편향으로 이어짐

### The origin of the modality gap.
- 마찬가지로 정보 불균형이 모달리티 갭의 주요 원인임
- 불균형은 임베딩 정렬을 제한하며, 이 경우 손실을 줄이기 위한 방법은 "uniformity"를 극대화 하는 것. 즉, negative pair의 거리를 증가시키는데 집중하게 되며, 이는 모달리티 갭을 초래함
- 저자는 이 과정이 그림 4에서 볼 수 있는 몇몇 차원을 사용하여 "uniformity"를 극대화하지만, 정렬에는 미미한 영향을 미친다고 말함

### Experimental validation in a fully-controlled synthetic setting.
![fig7](https://1drv.ms/i/c/af5642aec05791fb/IQRjInrzxlSyQYGS7eJqZjc-AZjH0LHZsA53qUO7L7kmIJA?width=949&height=734)
- MAD의 이미지는 그대로두고, 캡션에 포함된 속성의 수를 변경하여, 정보 불균형을 조작한 실험을 수행
- 그림 7(a)는 정보 불균형을 줄일 때, 모달리티 갭과 객체 편향이 모두 감소함을 보여주며,성능이 향상됨을 보여줌
- 그림 7(b)는 모달리티 갭이 모델 초기화시에 크더라도, Contrastive loss가 이를 줄일 수 있음을 보여주며, 기존의 원뿔효과 가설을 부정

### Experimental validation on real data.
- 실제 데이터에도 적용되는지 확인하기 위해, CC12M에서 3가지 설정(full caption, half caption, quarter caption)의 CLIP 모델 학습을 수행
    - full caption: 완전한 캡션
    - half caption: 앞/뒤 반으로 나누고, 둘 중에 하나를 랜덤으로 선택
    - quarter caption: 4등분하고, 하나를 랜덤으로 선택
- 그림 7(c)는 정보 불균형이 낮을수록(-> full caption) 모달리티 갭이 작아짐을 보여주며, 실제 데이터에서도 가설이 유효함을 확인할 수 있음

![fig15](https://1drv.ms/i/c/af5642aec05791fb/IQQIfk1k6NzrQ4EuFO7jlMaDASw8bjUdttVRd2i-hx8kRh4?width=944&height=749)
- (appendix, Revision)
- 리뷰어가 half, quarter이 문법적으로 맞는지 질문하였고, 저자는 MAD는 문법적으로 완전하나, 이와 달리 실제 데이터는 동일한 수준의 제어는 어렵다고 답변하였으며, 이전 연구에 따르면 CLIP이 Bag-of-words 표현을 학습하므로 미세한 문법 구조에 크게 의존하지 않는다고 답변
- 또한 추가적인 그림 15를 통해 단어가 아닌 문장을 drop하여 문법적으로 완벽한 캡션에서도 모달리티 갭은 같은 경향을 보임을 확인하였고, 성능 면은 큰 drop 전후 큰 차이가 없음

## 6.2. Is the Modality Gap a Bug or a Feature?

- 그림 7(c)는 full-caption과 half-caption의 학습 종료시 loss는 매우 유사하지만, R@1은 full-caption이 훨씬 높음을 보여줌
- 긍정적인 샘플을 잘 정렬하지 못함에도 불구하고, loss에 반영되지 않음을 저자는 logit의 엔트로피가 다르기 때문이라고 주장

### Entropy
-  VLM은 온도 매개변수를 통해서 (global) 엔트로피를 변경할 수 있지만, 모달리티 갭을 변경하는 것도 엔트로피를 변화시킬 수 있음
- 모달리티 갭을 통해 엔트로피를 변경하는 능력은, 모델이 각 샘플에 대해 엔트로피를 독립적으로 조절할 수 있는 유연성을 제공함, 이에 반해 온도 매개 변수는 엔트로피를 전역적으로만 영향을 미침

### Experimental validation.
![fig8](https://1drv.ms/i/c/af5642aec05791fb/IQS12vXRAyg9QZsaw-ZbwWkRAdgtzR1uOsP_9ljt71PTJ1U?width=933&height=201)
- CLIP이 엔트로피를 조정하기 위해 모달리티 갭을 변경하는지 조사하기 위해, CC12M에서 "Full caption"으로 학습 후, "quarter caption"(정보 불균형 증가로 엔트로피 증가) 으로 fine-tuning을 수행
- 두 방법의 엔트로피가 유사할 것으로 예상되며(그림 8(a)) 이에따라 온도 변수가 고정되었을 때, 모달리티 갭이 더 커져야함
- (그림 8(b-d)) 실제로 온도 변수를 고정하였을 때, 동일한 엔트로피를 맞추기 위해 모달리티 갭을 훨씬 더 증가시키는 것은 확인하였으며, 모달리티 갭은 버그가 아니라 기능으로 해석할 수 있음. 이는 모델이 엔트로피를 변경할 수 있는 유연성을 추가함
- "일정 수준의 모달리티 갭은 모델이 각 샘플 간 엔트로피를 조절하기 위해 필요할 수 있고, 그렇기 떄문에 버그가 아니라 기능으로 해석될 수 있다."고 이해하였음

### Review

- 모달리티 갭과 객체 편향이 발생하는 원인. 그리고 각각의 영향에 대해 심도있는 분석을 제공함 -> 학습에 쓰이는 텍스트가 중요하다..., caption enrichment도 저자가 언급하였음
- 직관적으로 모달리티 갭을 줄이는 것 만이 능사는 아니라고 생각하였는데, VLM을 설계할 때, 고려해야할 부분들을 많이 던져준 것 같음
- 이미지쪽도 이러한데, 정보 불균형이 더 큰 동영상은 이러한 편향이 더욱 심할것으로 예상이 되며, 제안한 지표로 모달리티 갭을 측정해볼 필요가 있을 것 같음
