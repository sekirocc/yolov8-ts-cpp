#include <torch/script.h>  // 包含 LibTorch

#include <opencv2/opencv.hpp>

int main() {
    // 加载保存的 TorchScript 模型
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("data/yolov8m.torchscript");
    } catch (const c10::Error& e) {
        std::cerr << "加载模型失败\n";
        return -1;
    }

    // 构造输入张量，4 张图片
    std::vector<torch::Tensor> tensors;

    // 图片文件路径
    std::vector<std::string> imagePaths = {
        "data/test_images/test_image_2_person.jpeg",
        "data/test_images/test_image_5_person.jpeg",
    };

    for (const auto& path : imagePaths) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "加载 img 失败" << path;
            return -1;
        }
        auto tensor = matToTensor(img);
        tensors.push_back(tensor);
    }

    // 合并成一个
    torch::Tensor batch_tensor = torch::stack(tensors);

    // 进行推理
    std::vector<torch::IValue> inputs{batch_tensor};
    at::Tensor output = module.forward(inputs).toTensor();

    auto sizes = output.sizes();
    std::cout << "sizes: " << sizes << std::endl;

    int batch_size = sizes[0];
    int max_detections = sizes[1];
    std::cout << "batch_size: " << batch_size << ", max_detection" << std::endl;

    // 显示推理后的结果
    for (int i = 0; i < batch_size; i++) {
        std::cout << "Results for image " << i << ":" << std::endl;
        for (int j = 0; j < max_detections; j++) {
            auto detection = output[i][j];
            float confidence = detection[4].item<float>();

            // 只处理置信度大于零的检测结果，如果图片没有足够的目标，detection
            // 元素的置信度会设置为 0
            if (confidence > 0) {
                float x_center = detection[0].item<float>();
                float y_center = detection[1].item<float>();
                float width = detection[2].item<float>();
                float height = detection[3].item<float>();
                float confidence = detection[4].item<float>();
                int class_id = static_cast<int>(detection[5].item<float>());

                std::cout << "  Detection " << j << ":" << std::endl;
                std::cout << "    Class ID: " << class_id << std::endl;
                std::cout << "    Confidence: " << confidence << std::endl;
                std::cout << "    Bounding Box (center): (" << x_center << ", "
                          << y_center << ", " << width << ", " << height << ")"
                          << std::endl;
            }
        }
    }

    return 0;
}
