# fsd_diff_npu

이 문서는 fsd_diff_npu 저장소의 안내문입니다. 이 저장소는 테슬라 FSD v15의 리얼리티 뷰(Reality View)를 조사하고, 라즈베리파이 같은 엣지 장치에 NPU(Hailo, DEEPX 등)를 붙여 비슷한 기능을 만드는 프로젝트를 담고 있습니다.

## 문서

- [FSD v15 리얼리티 뷰 기술 조사](docs/01_fsd_v15_reality_view_research.md): 처리 과정, 사용 모델, HW3, AI4, AI5 칩의 구조와 성능, 카메라 사양
- [테슬라 3B 주행 모델과 1B 확산 렌더러](docs/09_tesla_3b_1b_models.md): 공개된 내용, 테슬라 렌더러 규모 역산, 데이터와 학습 연산량과 엣지 배포 면에서 직접 만들 수 있는지
- [축소판 생성형 리얼리티 뷰 구현 계획](docs/10_mini_reality_view_build_plan.md): 공유 인코더, 작은 오토인코더, 한 단계 생성기를 파이 5와 DX-M1에 올리는 G1~G6 단계와 완료 기준
- [엣지 NPU 사양 비교](docs/02_edge_npu_spec_comparison.md): Hailo-8, 8L, 10H와 DEEPX DX-M1의 사양, 테슬라 AI4의 리얼리티 뷰 연산량과의 비교
- [테슬라 카메라 배치와 시야](docs/04_tesla_camera_layout.md): 카메라별 장착 위치, 방향, 화각, 최대 거리와 3D 도면([viz/tesla_camera_layout_3d.html](viz/tesla_camera_layout_3d.html))
- [테슬라 카메라 자리별 파이 5 카메라 선정](docs/06_pi5_camera_selection.md): 자리별 추천 제품, 화각과 HDR, 저조도 비교, 3대 이상 연결 방법, 국내 가격
- [차량 장착과 학습 순환 계획](docs/07_vehicle_mount_and_training_plan.md): 타공 없는 카메라 부착, 배선과 전원과 발열, 국내 법규, 테슬라의 학습 방식, 자동 라벨링, 추가 학습과 NPU 재배포 순환
- [엣지 NPU 구성과 모델 선정](docs/edge_npu_model_selection.md): 연산자 지원, BEV 모델 방식 비교, 단계별 계획
- [리얼리티 뷰 1단계 데모](docs/phase1_demo.md): 데모 코드의 구조, 설치와 실행, 카메라 보정
- [v15풍 화면](docs/v15_render.md): 3D 차량 모델, 그림자, 경로 띠, 시점 전환으로 v15 리얼리티 뷰를 닮게 그리는 화면과 자차 모델 교체 방법
- [라즈베리파이 5, Hailo-8, DX-M1 구현 계획](docs/03_pi5_hailo8_dxm1_implementation_plan.md): 두 NPU 연결 방법, 카메라 수와 배치, NPU별 모델 배치, 단계별 구현 순서와 완료 기준
- [진행 상황](docs/progress.md): 지금 단계, 측정값, 다음에 할 일, 사용자가 할 일, 크레딧 기록
- [Claude API 크레딧 사용 계획](docs/05_claude_api_credit_plan.md): 크레딧을 쓸 작업, 작업별 모델과 비용, 200달러 예산안, Console 설정과 호출 예시

자차 3D 모델은 Doz3의 "Kia Carnival 2011"(CC-BY-4.0)을 바꿔 쓴 것이고, 출처 표기는 [assets/vehicles/README.md](assets/vehicles/README.md)에 있습니다.
