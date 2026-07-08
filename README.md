# AUV ROS 2 Workspace

Bu repository, Teknofest deniz altı aracı için geliştirilmiş ROS 2 autonomy stack'i içerir.
Proje; pipe following, anomaly handling ve coordinate-based navigation akışlarını tek bir workspace içinde toplar.

## Project Highlights

- Modüler ROS 2 mimarisi
- Pipe segmentation ve pipe detection tabanlı takip akışı
- Low-level movement control ve heading/depth kontrol katmanı
- Mission state ve arm state yönetimi
- Coordinate-based navigation desteği

## Benim Katkım

Bu repository üzerinden özellikle aşağıdaki alanlar öne çıkarılabilir:

- ROS 2 system setup
- Node orchestration ve topic flow tasarımı
- Movement / control algorithms
- Mission state management
- Pipe following ve coordinate navigation tarafındaki kontrol mantığı

AI ve image processing node'ları ekip çalışmasının bir parçasıdır; ancak bu repository üzerinden asıl görünür olan katman, ROS 2 entegrasyonu ve kontrol tarafıdır.

## Repository Yapısı

- `pipe_segmentation` - camera frames üzerinden pipe segmentation yapar ve `/pipe_mask` yayınlar
- `pipe_detection` - mask üzerinden pipe pose error ve visibility çıkarır
- `movement_control` - pipe pose'u low-level motion commands'e çevirir
- `anomaly_detection` - camera stream'i izler ve mission-stop signal üretir
- `navigation_control` - vehicle'ı target coordinates'e yönlendirir
- `manual_control` - keyboard control ve MAVLink bridge içerir
- `joystick_control` - joystick-to-`/cmd_vel` bridge
- `rov_mode_manager` - lifecycle-based mode ve arm state management yapar

## How It Works

Sistemin akışı kabaca şu şekildedir:

1. Camera stream `/image_raw` üzerinden alınır.
2. `pipe_segmentation` bu görüntüden pipe mask üretir.
3. `pipe_detection` mask üzerinden pipe pose error ve visibility çıkarır.
4. `movement_control` bu bilgiyi low-level motion command'lere çevirir.
5. `cmd_vel_listener` bu komutları MAVLink tarafına taşır.
6. `anomaly_detection` ve `end_of_pipe_node` görev sonu / anomali durumlarını izler.
7. `rov_mode_manager` mod ve arm state yönetimini yapar.
8. `navigation_control` gerekirse coordinate-based navigation akışını yürütür.

Bu yapı, pipe following ve coordinate navigation gibi iki ana kullanım senaryosunu aynı ROS 2 workspace içinde gösterir.

## Setup

```bash
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

Eğer temiz bir ortamda başlıyorsan:

1. ROS 2 Humble kurulu olsun.
2. Bu workspace'i ROS 2 environment içine yerleştir.
3. `colcon build` çalıştır.
4. `source install/setup.bash` ile workspace'i aktif et.

## Run

Her package, `ros2 run` ile çalıştırılabilen bir executable sunar.

Örnek:

```bash
ros2 run pipe_segmentation pipe_segmentation_node
ros2 run pipe_detection pipe_detection_node
ros2 run movement_control movement_target_node
ros2 run anomaly_detection anomaly_detection_node
ros2 run navigation_control navigation_control_node
ros2 run manual_control manual_control_node
ros2 run manual_control cmd_vel_listener
ros2 run joystick_control joystick_control_node
ros2 run rov_mode_manager mode_manager_node
```

## Demo / Screenshots

Bu bölümde projeyi inceleyen kişiye hızlı bir görsel özet verilebilir:

- `rqt_graph` ile node graph
- `rqt_image_view` ile camera / mask / annotated image akışı
- Terminal log'ları ile mission flow
- Pipe following ve navigation davranışını gösteren kısa ekran görüntüleri

İstersen bu repo'ya sonradan `assets/` klasörü ekleyip burada gerçek görselleri paylaşabilirsin.

## Notes

- Bazı nodes, model weights veya hardware-specific ROS topics'e bağımlıdır.
- Bu repository'nin amacı, projede kullanılan gerçek source code ve autonomy logic'i göstermektir.
- `pipe_segmentation` ve `anomaly_detection` node'ları için model weights gereklidir.
- `end_of_pipe_node` için `pipe_mask` akışının çalışıyor olması gerekir.
- `manual_control` normal modda MAVLink connection bekler.

## Requirements

- ROS 2 Humble
- Python 3.10+
- OpenCV
- `cv_bridge`
- `pymavlink`
- `torch`
- `ultralytics`
- `segmentation_models_pytorch`
- `albumentations`
- `numpy`
- `rclpy`

## Dependencies

Bu repository'nin sağlıklı çalışması için temel bağımlılıklar:

- ROS 2 packages: `rclpy`, `std_msgs`, `geometry_msgs`, `sensor_msgs`, `cv_bridge`, `lifecycle_msgs`
- Python packages: `numpy`, `opencv-python` / `python3-opencv`, `torch`, `ultralytics`, `segmentation_models_pytorch`, `albumentations`, `pymavlink`
- Optional GUI tools: `rqt_graph`, `rqt_image_view`, `terminator`

Model tabanlı nodes için ek olarak:

- `src/pipe_segmentation/models/`
- `src/pipe_detection/models/`
- `src/anomaly_detection/models/`

## License

Apache-2.0
