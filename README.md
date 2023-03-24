# python-project6 [make_translator]  
## <한영번역기, Simple Seq2Seq and Seq2Seq with Attention>  
  
이번에 한영번역기를 만들어보았다.(사실 한거라고는 85% 완성된거에 15% 붙인거지만... )  
최종적으로는 팀원 당 각각 하나의 ipynb파일이 올라갔지만,  
이걸 하기위해 모두 각각 여러개 ipynb파일을 만들어서 여러가지 실험을 해봤다.  
여기에는 그 실험했던 부분을 정리해서 적을 예정이다. 또한 향후 ipynb파일을 모듈화해서 깃헙에 추가로 올릴 예정이다.  

-------------------------------------------------------------------------------------------------------------------------------------------------------
  
  
## <승훈>  
  
  
<import한 파일>  
한국어-영어 번역(병렬) 말뭉치 중 3개 파일(구어체(1), 구어체(2), 대화체) 총 50만 문장.  
[출처 : AI Hub(https://aihub.or.kr/)]  
* 여기서 데이터를 받으려면 가입 후 인증 받아서 신청해야함.  
  다양한 데이터가 있으므로 둘러보는것도 좋을 듯.  
  
  
### Preprocessing 및 tokenizing 
  
  
사실 AIhub의 대부분의 파일들은 문장이 깔끔하게 정렬도 되어있고 전처리할 필요가 거의 없으나,  
(가끔 맞춤법 틀린 문장이 간혹 보여 그것까지 고쳐주려했으나 여기서는 진행하지 않았다.)  
번역시에 반영이 안될수도 있거나 깨질수도 있는 문장이 있을 수 있어 그거를 전처리하는데 집중하였다.  
  
  
**1. unicode_hangeul 함수**
  
  
<img width="1101" alt="스크린샷 2023-03-23 오후 8 09 50" src="https://user-images.githubusercontent.com/121400054/227186172-3f155482-7efc-4139-9a30-9d94dedfa917.png">  
  
  
한글을 뗐다 붙이는 작업을 한다.  
가끔 한글이 예를 들어 '한글'이 아니라 'ㅎㅏㄴㄱㅡㄹ'이렇게 보이거나,   
설령 '한글'로 잘 보이더라도 막상 문자열 길이(len)를 출력하면 2가 아닌 6이 나오는 경우가 있다.  
그런 경우 뒤에 time_step이나 vocab_size에 영향을 줄 수 있어, 글자를 쪼갰다가 붙이는 작업을 함수로 작성했다.   
NFKC가 아니라 다른 방법도 있으니 찾아보면 좋다.  
  
  
**2. preprocess_sentence 함수**  
  
  
<img width="790" alt="스크린샷 2023-03-23 오후 8 14 55" src="https://user-images.githubusercontent.com/121400054/227188166-7a4f8f91-dcdf-4d40-8bf2-6aa13862c945.png">  
  
  
한글 전처리 함수. 구두점 사이에 공백을 만드는 이유는, 구두점을 토큰으로 분류하여 문장의 경계를 알 수 있도록 하기 위함이다.  
(출처 : https://wikidocs.net/21698)  
그 후 번역이 될 숫자, 영어, 한글, 몇개의 특수문자 등을 제외하고는 정규표현식으로 처리해준다.  
마지막에 lower 객체를 불러오는 이유는, 대문자를 없앰으로써 향후 토크나이저에 피팅시 차원을 축소하기 위함이다.  
  
  
**3. 토크나이저**  
  
  
이 부분에서 사실 상당히 고민했다.  
어떻게 문장을 잘라야 잘 인식을 해서 벡터로 변환이 되고, 그 받은 벡터를 잘 인식해서 다시 텍스트로 변환될지 고민이 됐기 때문이다.  
이 부분에서 3가지의 토크나이저를 이용해보았다.(텐서플로 토크나이저 포함하면 4가지)  
최종적으로는 한글은 HuggingFace Tokenizer를 이용해서 자른 다음 텐서플로 토크나이저에 fit하였고,  
영어는 그냥 공백을 기준으로 자른 다음 텐서플로 토크나이저에 fit하였다.  
  
  
아, 한글은 왜 그냥 공백 기준으로 안잘랐냐면,   
텐서플로 토크나이저에 fit했을 때 단어뭉치가 20만 문장기준 19만개로 어마어마하게 많이 나왔기 때문이다. 영어 문장은 단어 뭉치가 2만개였던거에 비해서.  
  
  
**3-1. 첫 번째 시도 // 한글 - mecab tokenizer, 영어 - nltk tokenizer**  
  
  
<img width="1135" alt="스크린샷 2023-03-23 오후 8 53 54" src="https://user-images.githubusercontent.com/121400054/227196058-21148dcc-08e0-4f66-80a7-7f0a368197dd.png">  
  
  
첫 번째로 한글은 mecab으로, 영어는 nltk로 토큰화를 해봤다. 보기에는 상당히 잘 토큰화가 된것 같다. 하지만,  
  
  
<img width="1315" alt="스크린샷 2023-03-23 오후 8 58 05" src="https://user-images.githubusercontent.com/121400054/227197039-b543b02b-7641-4606-a13c-d620dadbbf91.png">  
  
  
이렇게 인덱스화 한것을 보면 형태소로 상당히 잘게 끊겨들어간 것을 볼 수 있다.  
이는 단어 뭉치의 수에도 영향을 미치고, 최종적으로 벡터화했을 때 점수에도 영향을 미칠 것이다.  
참고로, 이렇게 토큰화 한 후 단어 뭉치의 수는 대화체 파일 10만 문장 기준 한글 21090개, 영어 16242개였다.  
  
  
<img width="335" alt="스크린샷 2023-03-23 오후 9 00 18" src="https://user-images.githubusercontent.com/121400054/227197583-b92a4047-1d5b-4669-a739-c81ae611e573.png">  
  
  
위 단어 뭉치를 이용해 번역한 BLEU 스코어 결과.  
물론 이건 Simple seq2seq를 이용했고, 10만 문장기준 10에폭 밖에 안돌렸지만 그래도 살짝 떨어진다.  
  
  
**3-2. 두 번째 시도 // 한글 - huggingFace, 영어 - huggingFace**  
  
  
자연어 처리 스타트업 허깅페이스가 개발한 패키지.  
huggingFace는 자주 등장하는 서브워드들을 하나의 토큰으로 취급하는 다양한 서브워드 토크나이저를 제공한다.  
(출처 : https://wikidocs.net/99893)  
특히 이 토크나이저는 구글에서 공개한 BERT의 토크나이저를 직접 구현한 BertWordPieceTokenizer가 있다.  
나도 이것을 사용했다.
  
  
이 토크나이저의 특징은 토크나이저를 먼저 학습을 시켜줘야한다는 점이다. 학습을 시켜서 자주 등장하는 단어의 서브워드를 파악한다.  
  
  
<img width="1092" alt="스크린샷 2023-03-23 오후 9 13 46" src="https://user-images.githubusercontent.com/121400054/227200819-2c9021de-5f5c-4c9a-94ad-8ed8e1d95694.png">  
  
  
이렇게 train을 시켜줘야한다. train의 주체는 당연히 우리가 학습시킬 문장들이다.  
여기서 vocab_size는 단어 집합의 크기, limit_alphabet : 병합 전의 초기 토큰의 허용 개수,  
min_frequency : 최소 해당 횟수만큼 등장한 쌍(pair)의 경우에만 병합 대상으로 정해줄 수 있다.   
나는 5번 미만으로 등장한 단어는 여기 반영하지 않았다.  
병합 전 초기 토큰은 어차피 0개였기 때문에, 나머지는 하이퍼파라미터로 정해줬다.  
  
  
<img width="1083" alt="스크린샷 2023-03-23 오후 9 05 31" src="https://user-images.githubusercontent.com/121400054/227198811-e01e2df1-dd81-45ab-a117-1030e1d5ccb8.png">  
  
  
이것을 사용한 결과. 형태소 단위로 끊은게 아니라 위에 설명한 대로 자주 등장한 어휘들을 하나로 끊어준게 돋보인다.  
특히 얘는 과거형도 끊어준다. 참고로 ##는 이 프로그램으로 토큰화할때 붙는 문자열인데,  
나중에 이 토크나이저를 통해 decode할 수도 있다.  
그 때 문장을 제대로 다시 변환하기위해 사용되는 문자열이다.  
물론 나는 여기서 그걸 안썼다. 그래서 번역 후에도 ##가 붙어나왔다...  
  
  
둘 다 이것을 썼을 때의 문제는 단어 뭉치 크기가 서로 생각보다 차이가 난다는 점이다.  
20만 문장 기준 한글 단어 뭉치의 크기는 44473개, 영어 단어 뭉치의 크기는 19655개였다.  
그래서, 디코딩 시에 반영이 안되는 단어가 많아졌을 것이다.  
  
  
<img width="1068" alt="스크린샷 2023-03-23 오후 9 10 47" src="https://user-images.githubusercontent.com/121400054/227199908-00cf3352-f87a-4647-b0e8-afd1073b5658.png">  
  
  
위 토크나이저로 번역한 결과. 보는 것과 같이 ##가 붙어 나온다.  
그래도 번역이 꽤 잘됐다. 40만 문장을 20만 문장 따로, 20만 문장 따로하여 각각 10에포크 씩 20에폭을 돌렸는데,
같은 구어체 파일을 fitting해서 그런지 model이 나중에 학습한 20만 문장에 편향됐음에도 불구하고 꽤 잘 나왔다.
  
  
<img width="308" alt="스크린샷 2023-03-23 오후 9 42 56" src="https://user-images.githubusercontent.com/121400054/227207204-f2170192-500a-44b0-9aa5-5ec216fac027.png">  
  
  
이것으로 번역한 결과의 BLEU 스코어는 0.07이었다. 이 결과는 어텐션 모델을 사용했고,   
학습을 좀 더 많이 시켜줘서 나오지 않았나 생각한다.  
  
  
**3-3. 최종 : 한글 - huggingface / 영어 - 공백 스플릿**    
  
  
결국 최종적으로 선택한 것은 한글은 huggingface(Kobert)를 쓰되, 영어는 공백으로만 잘라주는 것이었다.  
이렇게 했을 때 한글 문장의 각 단어 의미도 살면서, 영어 문장 단어도 보존이 잘 됐기 때문이다.  
추가로 번역을 했을 때 영어 문장에 ##이 안붙어서 더 깔끔하게 볼 수 있었다.  
다만 얘는 조금 치명적인(?) 단점이 있는데, 코랩같은 경우 런타임을 끊었다 다시 시작한 뒤 토크나이저를 다시 학습시키면  
단어 뭉치의 개수가 1~3개 정도 바뀐다. 만약 체크포인트 파일과 단어뭉치수가 안맞는다면, 맞춰줘야한다...  
  
  
<img width="1078" alt="스크린샷 2023-03-23 오후 9 37 02" src="https://user-images.githubusercontent.com/121400054/227205791-9017fe05-1b31-46c8-b8f9-952b084e1935.png">  
  
  
이처럼 문장이 깔끔하게 잘 토큰화 된 것을 볼 수 있다.  
이렇게 토큰화를 하고 단어 뭉치를 만든 결과, 50만 문장 기준 한글 단어뭉치는 57923개, 영어 단어뭉치는 60886개였다.  
huggingFace의 vocab_size는 60000, limit_alphabet은 10000, min_frequency는 5였다.  
  
  
최종 모델의 BLEU 스코어 결과는 아래에서 살펴보겠다.  
  
  
### translator modeling(encoder, decoder)  
  
  
번역기 모델은 encoder, decoder를 나누고, attention을 적용해줬다. 특히 decoder부분에 attention을 구현하려 노력했다.  
  
  
```python
# 인코더
# input, layer
encoder_inputs = Input(shape = (MAX_ENC_LEN,))
enc_emb_layer = Embedding(SRC_VOCAB_SIZE, EMBEDDING_DIM, name='ENC_Embedding')
enc_dropout = Dropout(0.2, name='ENC_Dropout')
enc_lstm = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True, name='ENC_LSTM')

# graph
enc_emb = enc_emb_layer(encoder_inputs)
enc_emb = enc_dropout(enc_emb)
encoder_outputs, enc_h, enc_c = enc_lstm(enc_emb)
encoder_states = [enc_h, enc_c]
```
  
  
encoder부분은 attention을 구현하기 위해 lstm layer에 return_seqeunce=True로 변경하여,  
누적된 Hidden state와 마지막 시점의 hidden state, cell state를 받아왔다.  
  
  
```python
# 디코더
# input, layer
decoder_inputs = Input(shape = (MAX_DEC_LEN,))
dec_emb_layer = Embedding(TAR_VOCAB_SIZE, EMBEDDING_DIM, name='DEC_Embedding')
dec_dropout = Dropout(0.2, name='DEC_Dropout')
dec_lstm = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True, name='DEC_LSTM')
att = Attention()
dense_tanh = Dense(HIDDEN_DIM, activation = 'tanh')
dec_dense = Dense(TAR_VOCAB_SIZE, activation='softmax', name='DEC_Dense')
dec_emb = dec_emb_layer(decoder_inputs)
dec_emb = dec_dropout(dec_emb)
decoder_output_, dec_h, dec_c = dec_lstm(dec_emb, initial_state=encoder_states)

# 어텐션을 구현해보자
# attention_score = tf.matmul(decoder_output_, encoder_outputs, transpose_b=True)
# attention_weight = tf.nn.softmax(attention_score)
# context_vector = tf.matmul(attention_weight, encoder_outputs)
# concat = Concatenate(axis=-1)([context_vector, decoder_output_])
# tanh_concat = dense_tanh(concat)
# decoder_outputs = dec_dense(tanh_concat)

# 어텐션 클래스를 사용해보자
context_vector = att([decoder_output_, encoder_outputs])
concat = Concatenate(axis=-1)([context_vector, decoder_output_])
tanh_concat = dense_tanh(concat)
decoder_outputs = dec_dense(concat)
```
  
  
나는 attention을 이렇게 표현하고자 하였다.  
  
  
1. st(decoder의 hidden state), ht(encoder의 hidden state들)를 활용해 attention score를 구한다.  
2. softmax를 활용해 Attention Distribution을 구한다.  
3. 인코더의 각 Attention Weight와 그에 대응하는 hidden state를 가중합하여 Attention Values를 구한다.  
4. Attention value와 decoder의 t 시점의 hidden state를 연결(concatenate)합니다.  
5. 출력층 연산의 input이 되는 st를 계산합니다.(tanh지남)  
6. 최종적인 예측 y^t를 얻습니다.  
  
  
직접 위처럼 구현해서 쓸수도 있으나, 쿼리와 키만 넣어주고 attention layer를 사용하면 위 3번 단계까지 계산한 결과를 리턴한다.  
그 이후 4번, 5번, 6번을 아래와 같이 구현하였다.  
5번의 경우 tf.math.tanh함수도 써봤는데, 그렇게 했더니 concat한 만큼의 dimention을 그대로 가지게 돼 에러가 나더라...  
에러를 스크린샷 찍고싶었는데 아쉽다.  
  
  
아, 하이퍼파라미터는 다음과 같이 설정하였다.  
BATCH_SIZE = 256, EPOCHS = 10, EMBEDDING_DIM = 128, HIDDEN_DIM = 256, NUM_SAMPLES = 500000  
이후에 모델을 구현하였다.
어텐션 layer를 적용한 model summary는 다음과 같다.
  
  
```python
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 50)]         0           []                               
                                                                                                  
 input_2 (InputLayer)           [(None, 64)]         0           []                               
                                                                                                  
 ENC_Embedding (Embedding)      (None, 50, 128)      7414144     ['input_1[0][0]']                
                                                                                                  
 DEC_Embedding (Embedding)      (None, 64, 128)      7793408     ['input_2[0][0]']                
                                                                                                  
 ENC_Dropout (Dropout)          (None, 50, 128)      0           ['ENC_Embedding[0][0]']          
                                                                                                  
 DEC_Dropout (Dropout)          (None, 64, 128)      0           ['DEC_Embedding[0][0]']          
                                                                                                  
 ENC_LSTM (LSTM)                [(None, 50, 256),    394240      ['ENC_Dropout[0][0]']            
                                 (None, 256),                                                     
                                 (None, 256)]                                                     
                                                                                                  
 DEC_LSTM (LSTM)                [(None, 64, 256),    394240      ['DEC_Dropout[0][0]',            
                                 (None, 256),                     'ENC_LSTM[0][1]',               
                                 (None, 256)]                     'ENC_LSTM[0][2]']               
                                                                                                  
 attention (Attention)          (None, 64, 256)      0           ['DEC_LSTM[0][0]',               
                                                                  'ENC_LSTM[0][0]']               
                                                                                                  
 concatenate (Concatenate)      (None, 64, 512)      0           ['attention[0][0]',              
                                                                  'DEC_LSTM[0][0]']               
                                                                                                  
 tanh_dense (Dense)                  (None, 64, 256)      131328      ['concatenate[0][0]']            
                                                                                                  
 DEC_Dense (Dense)              (None, 64, 60886)    15647702    ['dense[0][0]']                  
                                                                                                  
==================================================================================================
Total params: 31,775,062
Trainable params: 31,775,062
Non-trainable params: 0
__________________________________________________________________________________________________
```
  
    
attention을 안쓰고, attention을 직접 계산해서 모델을 구현하면 좀 더 직관적으로 어텐션이 어떻게 진행되는지 알 수 있다.  
아래는 attention을 직접 구현한 model summary다.  
  
  
```python
Model: "model_1"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 50)]         0           []                               
                                                                                                  
 input_3 (InputLayer)           [(None, 64)]         0           []                               
                                                                                                  
 ENC_Embedding (Embedding)      (None, 50, 128)      7413760     ['input_1[0][0]']                
                                                                                                  
 DEC_Embedding (Embedding)      (None, 64, 128)      7793408     ['input_3[0][0]']                
                                                                                                  
 ENC_Dropout (Dropout)          (None, 50, 128)      0           ['ENC_Embedding[0][0]']          
                                                                                                  
 DEC_Dropout (Dropout)          (None, 64, 128)      0           ['DEC_Embedding[0][0]']          
                                                                                                  
 ENC_LSTM (LSTM)                [(None, 50, 256),    394240      ['ENC_Dropout[0][0]']            
                                 (None, 256),                                                     
                                 (None, 256)]                                                     
                                                                                                  
 DEC_LSTM (LSTM)                [(None, 64, 256),    394240      ['DEC_Dropout[0][0]',            
                                 (None, 256),                     'ENC_LSTM[0][1]',               
                                 (None, 256)]                     'ENC_LSTM[0][2]']               
                                                                                                  
 tf.linalg.matmul (TFOpLambda)  (None, 64, 50)       0           ['DEC_LSTM[0][0]',               
                                                                  'ENC_LSTM[0][0]']               
                                                                                                  
 tf.nn.softmax (TFOpLambda)     (None, 64, 50)       0           ['tf.linalg.matmul[0][0]']       
                                                                                                  
 tf.linalg.matmul_1 (TFOpLambda  (None, 64, 256)     0           ['tf.nn.softmax[0][0]',          
 )                                                                'ENC_LSTM[0][0]']               
                                                                                                  
 concatenate_1 (Concatenate)    (None, 64, 512)      0           ['tf.linalg.matmul_1[0][0]',     
                                                                  'DEC_LSTM[0][0]']               
                                                                                                  
 tanh_dense (Dense)                (None, 64, 256)      131328      ['concatenate_1[0][0]']          
                                                                                                  
 DEC_Dense (Dense)              (None, 64, 60886)    15647702    ['dense_1[0][0]']                
                                                                                                  
==================================================================================================
Total params: 31,774,678
Trainable params: 31,774,678
Non-trainable params: 0
```
  
  
이렇게 모델을 구현한 후, 10에포크를 3번 반복, 총 30에포크 정도를 학습했다. 근데 사실 20에포크 이후로는 val_acc가 0.9이상으로는 안올라가더라...  
(ReduceLRonPleateu 적용했으면 좀 더 나았을지도...?)  
해당 사진은 10에폭을 2번 반복했을 때, 즉 20에폭 학습 이후의 사진이다.  
  
 <img width="711" alt="스크린샷 2023-03-23 오후 10 40 57" src="https://user-images.githubusercontent.com/121400054/227222121-7d2a390e-3af7-44ae-a77b-c81595ab223c.png">  
  
학습한 베스트 모델은 구글 드라이브에 checkpoint.h5로 저장하고, 모델이 바뀌지 않는 이상 언제든 연속으로 해당 체크포인트를 불러와 학습할 수 있게하였다.  
  
  
```python  
# 체크포인트로 현재 모델의 베스트 weight 저장  
checkpoint_path = '/content/drive/MyDrive/checkpoint.h5'  
checkpoint = ModelCheckpoint(filepath=checkpoint_path,   
                             save_weights_only=True,  
                             save_best_only=True,  
                             monitor='val_loss',   
                             verbose=1  
                            )  
# 연속하여 학습시 체크포인트를 로드하여 이어서 학습 
model.load_weights(checkpoint_path)  
```
  
  
이번엔 predict할 때 쓰기 위해, encoder_model과 decoder_model을 각각 설정해주는 부분이다.  
  
  
```python  
# 인코더(predict)  
encoder_model = Model(encoder_inputs, [encoder_outputs, encoder_states])  
```
  
  
encoder_model을 정의해줄 때, x값은 encoder_input으로, y값은 return_sequence=True한 값들이 그대로 나오도록 했다.  
  
  
```python
# 디코더(predict)

# Input Tensors : 이전 시점의 상태를 보관할 텐서
decoder_input_h = Input(shape=(HIDDEN_DIM,))
decoder_input_c = Input(shape=(HIDDEN_DIM,))

decoder_states_inputs = [decoder_input_h, decoder_input_c]

# 훈련 때 사용했던 임베딩 층을 재사용
x = dec_emb_layer(decoder_inputs)

# 다음 단어 예측을 위해 이전 시점의 상태를 현 시점의 초기 상태로 사용
x, state_h2, state_c2 = dec_lstm(x, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]

# 수정된 디코더
attn_layer = att([x, encoder_outputs])
decoder_concat = Concatenate(axis=-1)([attn_layer, x])
attn_out = dense_tanh(decoder_concat)
decoder_outputs = dec_dense(attn_out)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs + [encoder_outputs],
    [decoder_outputs] + decoder_states2)
```
  
  
decoder_model은 attention해준 값을 함께 넣어 정의해줬다.  
x값으로는 lstm모델에 반영될 decoder_input과 decoder_states_inputs,  
그리고 attention때 쓸 encoder_output을 함께 반영하였다.  
y값으로는 attention을 모두 거친 decoder_outputs와, lstm을 거친 마지막 decoder hidden, cell state를 함께 반영해주었다.  
  
  
```python
def translate(sentence):
    sentence = preprocess_sentence(sentence)
    tokens = hug_kor_tok.encode(sentence).tokens

    # 입력 문장 토큰 -> 라벨링
    enc_input = tokenizer_kor.texts_to_sequences([tokens])

    # 입력 문장 라벨링 -> 패딩 
    enc_input = tf.keras.preprocessing.sequence.pad_sequences(enc_input, maxlen=MAX_ENC_LEN, padding='post')
    encoder_output, states_value = encoder_model.predict(enc_input)

    # Decoder input인 <SOS>에 해당하는 정수 생성
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = tar2idx['<sos>']

    # prediction 시작
        # stop_condition이 True가 될 때까지 루프 반복
        # 구현의 간소화를 위해서 이 함수는 배치 크기를 1로 가정합니다.
    stop_condition = False
    decoded_sentence = ''

    for t in range(MAX_DEC_LEN):

        # 이전 시점의 상태 states_value를 현 시점의 초기 상태로 사용
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value + [encoder_output], verbose = 0)

        # 예측 결과를 단어로 변환
        result_token_index = np.argmax(output_tokens[0, -1, :])
        result_word = idx2tar[result_token_index]

        # 현재 시점의 예측 단어를 예측 문장에 추가
        decoded_sentence += ' ' + result_word

        # 현재 시점의 예측 결과 -> 다음 시점의 입력으로 업데이트
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = result_token_index

        # 현재 시점의 상태 ->  다음 시점의 상태로 업데이트
        states_value = [h, c]

        #  Stop condition <eos>에 도달하면 중단.
        if result_word == '<eos>':
            break 

    return decoded_sentence.strip('<eos>')
```
  
  
마지막으로 translate함수 부분이다. encoder_model에 우리가 번역하고싶은 문장(전처리 된)을 넣으면,  
output, state를 리턴하며 그 state와 encoder_output이 attention을 위해 함께 decoder_model에 반영되는 것을 볼 수 있다.  
  
  
### 번역 예시 및 BLEU score 결과  
  
  
번역 예시는 아래 사진과 같다. 나름 결과가 잘 나온다.
  
  
<img width="667" alt="스크린샷 2023-03-23 오후 10 42 22" src="https://user-images.githubusercontent.com/121400054/227222508-ec5ff9ec-bbaf-4f50-8879-cb5256523fe1.png">  
  
  
최종 BLEU score는 아래와 같다. (샘플은 50 문장이다.)   
  
  
<img width="352" alt="스크린샷 2023-03-23 오후 10 44 24" src="https://user-images.githubusercontent.com/121400054/227222918-04cfc691-69aa-4053-a4e4-db1fc2be7925.png">  
  
  
위와 비교하면 잘 나오는 것을 알 수 있다. 물론 3시간 학습한 번역기 성능이 어떻겠냐만은.  
게다가 인코더, 디코더 층도 한층 씩 밖에 안쌓았는데...  
  
  
### <승훈> 느낀 점
  
  
1. 행렬의 값을 계산하는 부분을 지속적으로 수업에서 배웠는데, 재밌었지만 왜 배우는 지는 사실 몰랐다.  
   근데 코드를 짜면서, 내가 직접으로 어느 부분에서 어떤 모양의 행렬이 나오는지를 계산해야 모델이 구현이 됐다.  
   그래서 행렬 계산이 어마어마하게 중요하다는 것을 알았다.  
  
  
2. 모든 학습은 코랩 유료버전에서 진행했다.  
   20만 문장의 경우는 gpu 스탠다드로 했으나,  
   50만 문장은 gpu용량이 커 프리미엄으로 진행했다.  
   이걸로 알 수 있듯, 학습에는 돈과 시간만 있으면 뭐든지 할 수 있다. ㅋㅋㅋㅋ  
   역시 돈으로 안되는게 없다. 하긴 챗GPT는 한번 학습하는데 100억원 든다는데 나는 한 번 학습하는데 1000원이면 뭐... 나쁘지 않은듯.  
   최종 이 모델의 학습을 하는데 약 2~3시간이 걸렸는데, 더 많은 데이터와 시간, 그리고 돈을 투자하면 진짜 멋있는 번역기가 나올 것 같다.  
  
  
3. 추가로 50만 문장을 더 학습시켜서(문어체) 100만 문장에 fit한 모델로 측정을 하기 위해  
   모델 변경 없이 데이터만 model.fit()을 이용해서 진행했는데도 불구, 데이터 편향 문제가 생겼다.  
   아마 구어체 학습한 것이 아닌 문어체를 학습하여 모델이 편향된 듯 했다.  
   그래서 찾아보니 continual learning에 관한 연구가 있었는데, 자율주행 간 도로상태 등을 지속적으로 업데이트하는데 이런 연구가 쓰인다고 한다.  
   나중에 한번 공부해봐야겠다는 생각을 가졌다.  
  
  
-------------------------------------------------------------------------------------------------------------------------------------------------------
  
## <준규>  

<import한 파일>  
구어체(1) 파일 / 20만 문장

### Preprocessing 및 tokenizing   
  
준규의 경우 한글에는 mecab, 영어는 텐서플로 토크나이저의 text_to_word_sequence를 사용하였다.
텐서플로 토크나이저의 text_to_word_sequence가 스플릿 및 소문자로도 자동으로 바꿔줬고, 끊어주는 것이 더 마음에 들었다고함.  
확인 결과, 진짜 깔끔하게 잘리긴 했다.  
  
<img width="960" alt="스크린샷 2023-03-24 오전 10 48 47" src="https://user-images.githubusercontent.com/121400054/227403782-db006398-6d67-4c4c-99f4-b493bf29b9b3.png">
  
  
이렇게 자른 결과 단어뭉치의 크기는 한국어가 37934개, 영어가 40004개 였다.  
  
  
### Translator Modeling (encoder, decoder)  
  
  
번역기의 경우는 단순 seq2seq로 구현하였다.  
  
  
```python

# 인코더 (Train)

## 층 
enc_emb_layer = Embedding(SRC_VOCAB_SIZE, EMBEDDING_DIM) # (201489, 64) SRC_VOCAB_SIZE는 한국어단어 집합의 크기
encoder_lstm = LSTM(HIDDEN_DIM, return_state = True) # 256, 상태값 리턴을 위해 return_state는 True


## input -> output 
encoder_inputs = Input(shape=(None, ))
enc_emb = enc_emb_layer(encoder_inputs)
encoder_outputs, state_h, state_c =  encoder_lstm(enc_emb)

encoder_states = [state_h, state_c] # 인코더의 은닉 상태와 셀 상태를 리스트로 저장
```
  
  
학습 인코더의 경우 return_sequence = False로, 마지막 층의 상태만 나오도록 했다.  
  
  
```python

# 디코더 (Train)
decoder_inputs = Input(shape=(None, ))

## 층
#상태값 리턴을 위해 lstm layer return_state는 True, 모든 시점에 대해서 단어를 예측하기 위해 return_sequences는 True
dec_emb_layer = Embedding(TAR_VOCAB_SIZE, HIDDEN_DIM)
decoder_lstm = LSTM(HIDDEN_DIM, return_sequences=True, return_state = True) 
decoder_dense = Dense(TAR_VOCAB_SIZE, activation = 'softmax')


## input -> output 
# 인코더 RNN의 은닉 상태를 초기 은닉 상태(initial_state)로 사용
dec_emb = dec_emb_layer(decoder_inputs)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state = encoder_states)

# 모든 시점의 결과에 대해서 소프트맥스 함수를 사용한 출력층을 통해 단어 예측
decoder_outputs = decoder_dense(decoder_outputs)
```
  
  
학습 디코더의 경우, 어텐션 층 없이 바로 RNN층과 softmax를 통해 결과를 예측하도록 했다.
model summary는 다음과 같다.
  
  
```python
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, None)]       0           []                               
                                                                                                  
 input_2 (InputLayer)           [(None, None)]       0           []                               
                                                                                                  
 embedding (Embedding)          (None, None, 64)     2427776     ['input_1[0][0]']                
                                                                                                  
 embedding_1 (Embedding)        (None, None, 256)    10241024    ['input_2[0][0]']                
                                                                                                  
 lstm (LSTM)                    [(None, 256),        328704      ['embedding[0][0]']              
                                 (None, 256),                                                     
                                 (None, 256)]                                                     
                                                                                                  
 lstm_1 (LSTM)                  [(None, None, 256),  525312      ['embedding_1[0][0]',            
                                 (None, 256),                     'lstm[0][1]',                   
                                 (None, 256)]                     'lstm[0][2]']                   
                                                                                                  
 dense (Dense)                  (None, None, 40004)  10281028    ['lstm_1[0][0]']                 
                                                                                                  
==================================================================================================
Total params: 23,803,844
Trainable params: 23,803,844
Non-trainable params: 0
```
  
  
훈련을 시킬 때 encoder_model과 decoder_model은 다음과 같이 정의하였다.  
  
  
```python
# 인코더 - 훈련과 동일
encoder_model = Model(encoder_inputs, encoder_states)

# 디코더 (Predict)
# Input Tensors : 이전 시점의 상태를 보관할 텐서
decoder_input_h = Input(shape=(HIDDEN_DIM,))
decoder_input_c = Input(shape=(HIDDEN_DIM,))

decoder_states_inputs = [decoder_input_h, decoder_input_c]

# 훈련 때 사용했던 임베딩 층을 재사용
x = dec_emb_layer(decoder_inputs)

# 다음 단어 예측을 위해 이전 시점의 상태를 현 시점의 초기 상태로 사용
x, state_h2, state_c2 = decoder_lstm(x, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]

# 모든 시점에 대해서 단어 예측 (Fully Connected)
x = decoder_dense(x)

# 수정된 디코더
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [x] + decoder_states2)
```
  
  
마지막으로 Translate함수는 아래와 같이 정의하였다.  
  
  
```python
ef translate(sentence):
    tokens = preprocess_korean_sentence(sentence)
    # tokens = sentence.split(" ")

    # 입력 문장 토큰 -> 라벨링
    enc_input = tokenizer_enc.texts_to_sequences([tokens])

    # 입력 문장 라벨링 -> 패딩 
    enc_input = tf.keras.preprocessing.sequence.pad_sequences(enc_input, maxlen=MAX_ENC_LEN, padding='post')
    states_value = encoder_model.predict(enc_input)

    # Decoder input인 <SOS>에 해당하는 정수 생성
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = tar2idx['<sos>']


    # prediction 시작
        # stop_condition이 True가 될 때까지 루프 반복
        # 구현의 간소화를 위해서 이 함수는 배치 크기를 1로 가정합니다.
    stop_condition = False
    decoded_sentence = ''

    for t in range(MAX_DEC_LEN):

        # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose = 0)

        # 예측 결과를 단어로 변환
        result_token_index = np.argmax(output_tokens[0, -1, :])
        result_word = idx2tar[result_token_index]

        # 현재 시점의 예측 단어를 예측 문장에 추가
        decoded_sentence += ' '+result_word

        # 현재 시점의 예측 결과 -> 다음 시점의 입력으로 업데이트
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = result_token_index

        # 현재 시점의 상태 ->  다음 시점의 상태로 업데이트
        states_value = [h, c]

        #  Stop condition <eos>에 도달하면 중단.
        if result_word == '<eos>':
            break 

    return decoded_sentence.strip(' <eos>')
```

### 번역 예시 및 BLEU score 결과  

위의 모델로 실행한 번역 결과와 예측 결과는 다음과 같다. 학습은 10에폭, Batch_size는 128로 진행하였다.  
  
  
<img width="760" alt="스크린샷 2023-03-24 오전 11 07 16" src="https://user-images.githubusercontent.com/121400054/227406239-94dbdcc4-1c1b-4b68-a84a-540ecb7ace2d.png">
  
  
<img width="311" alt="스크린샷 2023-03-24 오전 11 09 23" src="https://user-images.githubusercontent.com/121400054/227406457-acf2b6b8-dbc3-4c0d-b03e-5d27233e3954.png">


### <준규> 느낀 점  

1. konlpy morphs함수의 stem파라미터를 True로 주었을 때가 False로 주었을 때보다 BLEU score가 더 잘 나오는 경향이 있었으나,  
   과거형 그대로 번역해야 하는 문장의 뜻을 바꿔버리는 경향이 있음을 알 수 있었다.  

2. 또한 각종 조사와 불용어를 제거하여 직관적으로 번역하면 준수한 결과가 나올 것이라고 예상하여 한국어, 영어 stopword를 설정했으나, 
   오히려 BLEU score가 형편없게 나와버리는 경향이 있었다.  
   
-------------------------------------------------------------------------------------------------------------------------------------------------------

## <은수>  
  
  
<import한 파일>  
구어체(1) 파일 / 20만 문장
  
  
### Preprocessing 및 tokenizing   
  
  
은수의 경우 토크나이징에 한국어는 okt, 영어는 공백 스플릿 후 텐서플로 토크나이저로 시퀀스를 줬다.  
그리고 특이한 점으로, 25단어 이상의 문장은 제외하였다는 것이다.  
문장이 길어질 수록 결과가 잘 안나오기 때문에, 이렇게 처리를 하셨다고 한다.  
  
  
<img width="398" alt="스크린샷 2023-03-24 오후 12 37 59" src="https://user-images.githubusercontent.com/121400054/227418340-242f2253-2ae2-4ed0-8cb7-b5f940638c74.png">

  
위는 한국어, 영어 각 문장의 단어길이 분포를 보여주는 그래프들이다. 
  
  
<img width="619" alt="스크린샷 2023-03-24 오후 12 41 22" src="https://user-images.githubusercontent.com/121400054/227418774-fd336a23-958d-4d92-a8b3-9a25e2a20295.png">
  
  
전처리 및 25단어 이상을 가진 문장을 제외한 결과이다.  
한국어 단어 뭉치 수는 67370, 영어 단어 뭉치 수는 40879로, 20만 문장 치고는 상당히 단어 뭉치가 많다.
  
  
### Translator Modeling (encoder, decoder)  
  

모델의 경우, 하이퍼파라미터는 이렇게 설정하였다.  
emb_dim= 1000, hid_dim=256, bi_hid_dim=int(hid_dim/2)  
특이한 점으로 bi_hid_dim이라는 것이 있는데, 이는 bidirectional(양방향)LSTM 모델에 적용한 dimention이다.  
즉, bidirectional RNN 모델을 translator에 추가하였다.  
이는 luong et al.(2015)이 작성한,  
Effective Approaches to Attention-based Neural Machine Translation 논문에 있는 모델을 구현하기위해서였다고 한다.  
실제로 논문에 있는 모델에서는 최대 6가지의 기법을 활용하여, 총 8개 모델의 BLEU score를 보여준다.  
(Base + reverse + dropout + local attention + feed input + unk replace)  
논문에 있는 것들을 최대한 구현하려고 한게 돋보인다.  
  
  
```python
# 인코더(train)
enc_emb_layer = Embedding(SRC_VOCAB_SIZE, emb_dim)
encoder_bilstm = Bidirectional(LSTM(bi_hid_dim,dropout=0.2, return_sequences=True,return_state=True)) 

encoder_inputs= Input(shape=(None,))
encoder_emb= enc_emb_layer(encoder_inputs)
encoder_outputs, forward_h, forward_c, backward_h, backward_c  = encoder_bilstm(encoder_emb)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h,state_c]
```
  
  
encoder 모델에 bidirectional LSTM layer와, 
그 층을 통과한 결과로 forward hidden/cell state, backward hidden/cell state가 나온 것을 볼수 있다.  
이것을 각각 concat하여, 나중에 attention을 위해 encoder_states라는 변수에 지정하였다.  
  
  
```python
# 디코더(train)
dec_emb_layer = Embedding(TAR_VOCAB_SIZE, hid_dim)
decoder_lstm = LSTM(hid_dim, dropout=0.2,return_sequences=True, return_state=True)
decoder_dense1 = Dense(hid_dim,activation='tanh')
decoder_dense2 = Dense(TAR_VOCAB_SIZE, activation='softmax')

decoder_inputs = Input(shape=(None,))
dec_emb = dec_emb_layer(decoder_inputs)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state = encoder_states)
attention = layers.Attention()([decoder_outputs,encoder_outputs])
concat = Concatenate()([decoder_outputs,attention])
dense1 = decoder_dense1(concat)
decoder_outputs = decoder_dense2(dense1)
```
  
  
decoder모델은 bidierctional이 아닌 그냥 LSTM을 적용하였다. 내 모델과 다른 부분이 하나 있다면,   
내 모델은 Dropout층을 하나 아예 만들었고 은수님 모델은 LSTM에 dropout을 적용했다는 것이다.  
  
  
```python
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, None)]       0           []                               
                                                                                                  
 embedding (Embedding)          (None, None, 1000)   67370000    ['input_1[0][0]']                
                                                                                                  
 input_2 (InputLayer)           [(None, None)]       0           []                               
                                                                                                  
 bidirectional (Bidirectional)  [(None, None, 256),  1156096     ['embedding[0][0]']              
                                 (None, 128),                                                     
                                 (None, 128),                                                     
                                 (None, 128),                                                     
                                 (None, 128)]                                                     
                                                                                                  
 embedding_1 (Embedding)        (None, None, 256)    10465024    ['input_2[0][0]']                
                                                                                                  
 concatenate (Concatenate)      (None, 256)          0           ['bidirectional[0][1]',          
                                                                  'bidirectional[0][3]']          
                                                                                                  
 concatenate_1 (Concatenate)    (None, 256)          0           ['bidirectional[0][2]',          
                                                                  'bidirectional[0][4]']          
                                                                                                  
 lstm_1 (LSTM)                  [(None, None, 256),  525312      ['embedding_1[0][0]',            
                                 (None, 256),                     'concatenate[0][0]',            
                                 (None, 256)]                     'concatenate_1[0][0]']          
                                                                                                  
 attention (Attention)          (None, None, 256)    0           ['lstm_1[0][0]',                 
                                                                  'bidirectional[0][0]']          
                                                                                                  
 concatenate_2 (Concatenate)    (None, None, 512)    0           ['lstm_1[0][0]',                 
                                                                  'attention[0][0]']              
                                                                                                  
 dense (Dense)                  (None, None, 256)    131328      ['concatenate_2[0][0]']          
                                                                                                  
 dense_1 (Dense)                (None, None, 40879)  10505903    ['dense[0][0]']                  
                                                                                                  
==================================================================================================
Total params: 90,153,663
Trainable params: 90,153,663
Non-trainable params: 0
```
  
  
위 모델의 summary결과이다. 확실히 Embedding Dimention이 1000개에, bidirectional LSTM등을 쓰니 파라미터가 큰 것이 보인다.  


![그림](https://user-images.githubusercontent.com/121400054/227421046-e9103edb-d402-47d1-9a91-04bf9bbfdffd.png)


위 모델을 시각화한 그림이다. 아 나도 이거 한다는걸 까먹었네 ;;  
다음은 예측을 위한 Encoder 및 decoder모델 코드이다. 내 모델과 다른 부분은 없다.

```python

# 인코더(predict)
encoder_model = Model(encoder_inputs, [encoder_states,encoder_outputs])

# 디코더(predict)
decoder_input_h = Input(shape=(hid_dim,))
decoder_input_c = Input(shape=(hid_dim,))
decoder_encoder_outputs =Input(shape=(MAX_ENC_LEN,hid_dim,))

decoder_states_inputs = [decoder_input_h,decoder_input_c]

x= dec_emb_layer(decoder_inputs)

x, state_h2, state_c2 = decoder_lstm(x, initial_state=decoder_states_inputs)

decoder_states2 = [state_h2, state_c2]

attention = layers.Attention()([x,decoder_encoder_outputs])
concat = layers.Concatenate()([x,attention])
dense1 = decoder_dense1(concat)
x = decoder_dense2(dense1)


decoder_model = Model([decoder_inputs]+decoder_states_inputs+[decoder_encoder_outputs], [x]+decoder_states2)
```

다음은 translate함수의 코드이다. 이 부분도 다른 것은 없다.

```python
def translate(sentence):
    sentence = preprocesskr(sentence)
    enc_input = tokenizer_enc.texts_to_sequences([sentence])
    enc_input = pad_sequences(enc_input, maxlen=MAX_ENC_LEN, padding='post')
    states_value, output_value = encoder_model.predict(enc_input)
    target_seq = np.zeros((1,1))
    target_seq[0,0] = tar2idx['<sos>']

    stop_condition = False
    decoded_sentence = ''

    for t in range(MAX_DEC_LEN):
        output_tokens, h, c = decoder_model.predict([target_seq]+ states_value+[output_value], verbose=0)
        result_token_index = np.argmax(output_tokens[0,-1,:])
        result_word = idx2tar[result_token_index]
        decoded_sentence += ' '+result_word
        target_seq = np.zeros((1,1))
        target_seq[0,0] = result_token_index
        states_value = [h,c]
        if result_word == '<eos>':
            break

    return decoded_sentence.strip(' <eos>')
```

### 번역 예시 및 BLEU score 결과  
  
  
위 모델을 이용한 번역 예시 및 BLEU score는 다음과 같다. 학습은 11에폭을 하였다. batch_size는 128이다.  
확실히 층이 깊어서 그런가... 문장 데이터와 길이를 감안하고도 점수가 높다.  
  
  
<img width="563" alt="스크린샷 2023-03-24 오후 1 04 33" src="https://user-images.githubusercontent.com/121400054/227421719-88c7cce4-186f-4369-be84-4d5a5e85efab.png">
  
  
![image](https://user-images.githubusercontent.com/121400054/227437282-45931848-8512-4a8a-bafc-dc8af1aee022.png)



### <은수> 느낀 점  
  
  
1. train, test파일 BLEU score를 채점할 때, 무작정 랜덤으로 뽑는 것이 아닌,
   random.seed를 이용해서 살짝 fix해서 랜덤으로 뽑았으면 그 범위가 좀 더 좁혀져서 BLEU score가 좀 더 잘 나오지 않았을까 생각한다.
     
 
### <은수> Bonus  
  
  
추가로 은수 위 모델을 확장하여 다른 모델까지 구현을 하였다. reference는 역시 luong의 논문이다.  
여기서는 Embedding Dimention을 500으로 줄여서 설정하였다. 다른 하이퍼 파라미터는 같다.  

```python
# 인코더(train)
#two stacked lstm #two states
enc_emb_layer = Embedding(SRC_VOCAB_SIZE, emb_dim)
#encoder_lstm = LSTM(hid_dim, return_state=True)
encoder_lstm1 = Bidirectional(LSTM(bi_hid_dim,dropout=0.5, return_sequences=True,return_state=True))
encoder_lstm2 = Bidirectional(LSTM(bi_hid_dim,dropout=0.5, return_sequences=True,return_state=True))


encoder_inputs= Input(shape=(None,))
encoder_emb= enc_emb_layer(encoder_inputs)
encoder_outputs1, forward_h1, forward_c1, backward_h1, backward_c1 = encoder_lstm1(encoder_emb)
state_h1 = Concatenate()([forward_h1, backward_h1])
state_c1 = Concatenate()([forward_c1, backward_c1])
encoder_states1 = [state_h1,state_c1]
encoder_outputs, forward_h, forward_c, backward_h, backward_c  = encoder_lstm2(encoder_outputs1)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h,state_c]
```

Bidirectional LSTM을 여기서는 2개를 쌓은 것을 볼 수 있다.  
추가적으로, dropout을 0.5로 설정하였는데 논문에는 0.2로 되어있으나 다른 결과를 보기 위해 은수님이 변경하였다.  
  
  
```python
# 디코더(train)
dec_emb_layer = Embedding(TAR_VOCAB_SIZE, hid_dim)
decoder_lstm1 = LSTM(hid_dim, dropout=0.5,return_sequences=True,return_state=True)
decoder_lstm2 = LSTM(hid_dim, dropout=0.5,return_sequences=True, return_state=True)
decoder_dense1 = Dense(hid_dim,activation='tanh')
decoder_dense2 = Dense(TAR_VOCAB_SIZE, activation='softmax')

decoder_inputs = Input(shape=(None,))
dec_emb = dec_emb_layer(decoder_inputs)
decoder_outputs1, _, _= decoder_lstm1(dec_emb, initial_state = encoder_states1)
decoder_outputs, _, _ = decoder_lstm2(decoder_outputs1, initial_state = encoder_states)
attention = layers.Attention()([decoder_outputs,encoder_outputs])
concat = Concatenate()([decoder_outputs,attention])
dense1 = decoder_dense1(concat)
decoder_outputs = decoder_dense2(dense1)
```

여기서도 LSTM을 2개를 쌓았다. 논문과 비슷하게 구현하려고 노력하였다고 한다.
  
  
![그림2](https://user-images.githubusercontent.com/121400054/227423311-71555071-3816-4690-8551-ec103c3c0311.png)
  
  
모델 요약본은 다음과 같다. 멋있는데 어지럽다. ㅋㅋㅋㅋㅋ  
결과는 다음과 같다. 학습은 5에폭을 하였고, batch_size는 128로 똑같다.  
처음 모델보다는 결과가 안좋은데, 아마 충분히 학습했다면 더 좋은 결과가 나오지 않았을까 싶다.  
  
  
<img width="335" alt="스크린샷 2023-03-24 오후 1 19 33" src="https://user-images.githubusercontent.com/121400054/227423482-18e97c7d-709e-4c77-ab45-4b648f01efcc.png">


-------------------------------------------------------------------------------------------------------------------------------------------------------
  
  
이상으로 우리 팀원이 작성했던 모든 모델과 코드를 살펴보았다.  
전처리 방법에 따라, 그리고 학습 모델에 따라 결과가 달라지는 부분을 보고 많은 부분을 배웠으면 좋겠다.  
  
  
-------------------------------------------------------------------------------------------------------------------------------------------------------
## 추가  


1차 : 승훈 모델 - 학습 : 10에폭  
BLEU : 0.1567
  
  
2차 : 승훈 모델 - 학습 : 20에폭  
BLEU : 0.1351
  
  
3차 : 은수 모델(2번째) - 학습 : 10에폭  
BLEU : 0.122
  
  
4차 : 은수 모델(1번째) - 학습 : 10에폭  
BLEU : 0.113


5차 : 은수 모델(1번째) - 학습 : 11에폭  
BLEU : 0.121


6차 : 은수 모델(1번째) - 학습 : 11에폭(inference 코드 재실행)  
BLEU : 0.121

-------------------------------------------------------------------------------------------------------------------------------------------------------

