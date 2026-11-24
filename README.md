# fsd_diff_npu

이 문서는 fsd_diff_npu 저장소의 안내문입니다. 이 저장소는 테슬라 FSD v15의 리얼리티 뷰(Reality View)를 조사하고, 라즈베리파이 같은 엣지 장치에 NPU(Hailo, DEEPX 등)를 붙여 비슷한 기능을 만드는 프로젝트를 담고 있습니다.

## 문서

| 문서 | 내용 |
|---|---|
| [FSD v15 리얼리티 뷰 기술 조사](docs/01_fsd_v15_reality_view_research.md) | 처리 과정, 사용 모델, HW3, AI4, AI5 칩의 구조와 성능, 카메라 사양 |
| [엣지 NPU 사양 비교](docs/02_edge_npu_spec_comparison.md) | Hailo-8, 8L, 10H와 DEEPX DX-M1의 사양, 테슬라 AI4의 리얼리티 뷰 연산량과의 비교 |
| [테슬라 카메라 배치와 시야](docs/04_tesla_camera_layout.md) | 카메라별 장착 위치, 방향, 화각, 최대 거리와 3D 도면([viz/tesla_camera_layout_3d.html](viz/tesla_camera_layout_3d.html)) |
| [엣지 NPU 구성과 모델 선정](docs/edge_npu_model_selection.md) | 연산자 지원, BEV 모델 방식 비교, 단계별 계획 |
| [리얼리티 뷰 1단계 데모](docs/phase1_demo.md) | 데모 코드의 구조, 설치와 실행, 카메라 보정 |
| [라즈베리파이 5, Hailo-8, DX-M1 구현 계획](docs/03_pi5_hailo8_dxm1_implementation_plan.md) | 두 NPU 연결 방법, 카메라 수와 배치, NPU별 모델 배치, 단계별 구현 순서와 완료 기준 |
