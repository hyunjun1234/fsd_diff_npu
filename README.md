# fsd_diff_npu

테슬라 FSD v15 리얼리티 뷰(Reality View)를 조사하고, 라즈베리파이 같은 엣지 장치에 NPU(Hailo, DEEPX 등)를 붙여 비슷한 기능을 구현하는 프로젝트입니다.

## 문서

- [01. FSD v15 리얼리티 뷰 기술 조사](docs/01_fsd_v15_reality_view_research.md): 파이프라인, 사용 모델, HW3/AI4/AI5 칩 구조와 성능, 카메라 스펙
- [02. 엣지 NPU 사양 비교](docs/02_edge_npu_spec_comparison.md): Hailo-8/8L/10H, DEEPX DX-M1 공식 사양과 테슬라 AI4 리얼리티 뷰 예산 비교
- [엣지 NPU 구성과 모델 선정](docs/edge_npu_model_selection.md): 연산자 지원, BEV 모델 방식 비교, 단계별 계획
