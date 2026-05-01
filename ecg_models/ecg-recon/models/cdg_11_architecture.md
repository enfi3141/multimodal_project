# CDGS11 구조도

요청하신 스타일처럼, 먼저 상속 관계와 `forward()` 손실 흐름을 한 장에서 보이도록 구성했습니다.

## 구조1 - 클래스 상속 및 forward() 전체 흐름

```mermaid
flowchart TB
    subgraph S0[ ]
      C10[CDGS10] --> C11[CDGS11]
      L10[CDGS10Loss] --> L11[CDGS11Loss]
    end

    C11 -. 구조 완전 동일 .-> FWD
    L11 -. + 2 정규화 항 추가 .-> FWD

    P[pred\n[B, L, T]] --> SUPER
    T[target\n[B, L, T]] --> SUPER
    A[amplitude\n[B, N]] --> SUPER

    E[envelope_t\n[B, N, T]] --> NEW
    D[dipole_dir_t\n[B, N, 3, T]] --> NEW
    A2[amplitude\n[B, N]] --> NEW

    SUPER[super().forward()\nCDGS10Loss 기본 항 계산] --> BASE[total_base + terms]
    NEW[신규 정규화 2항 (CDGS11)] --> ENVL[l_env_decor\nw=0.01]
    NEW --> DIRL[l_dir_div\nw=0.005]

    BASE --> SUM
    ENVL --> SUM
    DIRL --> SUM

    SUM[total = total_base + 0.01*l_env_decor + 0.005*l_dir_div\nterms.update(env_decor, dir_div, total)]

    classDef gray fill:#4c4c4c,stroke:#8a8a8a,color:#ffffff;
    classDef purple fill:#4a3aa7,stroke:#7f73d9,color:#ffffff;
    classDef teal fill:#0b6b63,stroke:#24a596,color:#d9fff9;
    classDef blue fill:#15529b,stroke:#4a8bd9,color:#e7f2ff;
    classDef amber fill:#6b4300,stroke:#c5851a,color:#ffd89c;

    class C10,L10,SUPER,BASE gray;
    class C11,L11,P,T purple;
    class E,ENVL,NEW teal;
    class D,DIRL,A,A2 blue;
    class SUM amber;
```

## 구조2 - 신규 정규화 2항 내부 계산

```mermaid
flowchart LR
    subgraph ED[Envelope Decorrelation]
      E0[envelope_t] --> E1[time stride: env_time_stride]
      E1 --> E2[top-k 슬롯 선택\n기준: amplitude]
      E2 --> E3[slot별 평균 제거]
      E3 --> E4[L2 normalize]
      E4 --> E5[sim = env * env^T]
      E5 --> E6[offdiag abs mean]
      E6 --> E7[l_env_decor]
    end

    subgraph DD[Direction Diversity]
      D0[dipole_dir_t] --> D1[time stride: dir_time_stride]
      D1 --> D2[top-k 슬롯 선택\n기준: amplitude]
      D2 --> D3[방향축 normalize]
      D3 --> D4[시간축 펼침: (B*T, K, 3)]
      D4 --> D5[sim = dirs * dirs^T]
      D5 --> D6[offdiag abs mean]
      D6 --> D7[l_dir_div]
    end

    classDef teal fill:#0b6b63,stroke:#24a596,color:#d9fff9;
    classDef blue fill:#15529b,stroke:#4a8bd9,color:#e7f2ff;
    class E0,E1,E2,E3,E4,E5,E6,E7 teal;
    class D0,D1,D2,D3,D4,D5,D6,D7 blue;
```

## 구조3 - 최종 수식

```text
L_cdgs11 = L_cdgs10 + w_env_decor * L_env_decor + w_dir_div * L_dir_div

default:
  w_env_decor = 0.01
  w_dir_div   = 0.005
  decor_top_k = 64
  env_time_stride = 4
  dir_time_stride = 8
```

## 빠른 체크 포인트

- CDGS11 본체는 CDGS10과 동일 (모델 구조 변경 없음)
- 변경점은 CDGS11Loss의 두 정규화 항 추가
- 최종 반환은 `(total, terms)`이며 `terms`에 `env_decor`, `dir_div`, `total`이 추가됨
