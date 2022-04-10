// tensorflow-c.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <array>
#include <vector>
#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>

void example_sr()
{
    /* https://int-i.github.io/cpp/2020-11-18/libtensorflow/ */
    // Show version information of the tensorflow
    std::cout << "TensorFlow Version: " << TF_Version() << std::endl;

    // Do initializations for tensorflow session
    auto* run_options = TF_NewBufferFromString("", 0);
    auto* session_options = TF_NewSessionOptions();
    auto* graph = TF_NewGraph();
    auto* status = TF_NewStatus();
    std::array<char const*, 1> tags{ "serve" };

    // Load the tensorflow model
    const std::string model_path = "../../sr/EDSR";
    auto* session = TF_LoadSessionFromSavedModel(session_options, run_options,
        model_path.c_str(), tags.data(), tags.size(),
        graph, nullptr, status);
    if (TF_GetCode(status) != TF_OK) {
        std::cout << TF_Message(status) << '\n';
    }

    // Define network operations
    const std::string input_op_name = "serving_default_img";
    auto* input_op = TF_GraphOperationByName(graph, input_op_name.c_str());
    if (input_op == nullptr) {
        std::cout << "Failed to find graph operation\n";
    }
    const std::string output_op_name = "StatefulPartitionedCall";
    auto* output_op = TF_GraphOperationByName(graph, output_op_name.c_str());
    if (output_op == nullptr) {
        std::cout << "Failed to find graph operation\n";
    }
    std::array<TF_Output, 1> input_ops = { TF_Output{ input_op, 0 } };
    std::array<TF_Output, 1> output_ops = { TF_Output{ output_op, 0 } };

    // Load an input image
    cv::Mat lr = cv::imread("../../sr/cifar10_lr1.jpg");            // low resolution
    cv::Mat hr = cv::imread("../../sr/cifar10_hr1.jpg");            // high resolution
    lr.convertTo(lr, CV_32F);
    hr.convertTo(hr, CV_32F);

    // Set input data
    std::array<std::array<std::array<float, 3>, 32>, 32> x;
    float* lr_ptr = reinterpret_cast<float*>(lr.data);
    for (int i = 0; i < lr.rows; i++)
    {
        for (int j = 0; j < lr.cols; j++)
        {
            for (int c = 0; c < lr.channels(); c++)
            {
                x[i][j][c] = lr_ptr[i * lr.cols * lr.channels() + j * lr.channels() + c];
            }
        }
    }
    std::vector<std::array<std::array<std::array<float, 3>, 32>, 32>> inputs{ x };
    std::array<int64_t, 4> const dims{ static_cast<int64_t>(inputs.size()), 32, 32, 3 };
    void* data = (void*)inputs.data();
    std::size_t const ndata = inputs.size() * 32 * 32 * 3 * TF_DataTypeSize(TF_FLOAT);
    auto const deallocator = [](void*, std::size_t, void*) {}; // unused deallocator because of RAII
    auto* input_tensor = TF_NewTensor(TF_FLOAT, dims.data(), dims.size(), data, ndata, deallocator, nullptr);
    std::array<TF_Tensor*, 1> input_values{ input_tensor };
    std::array<TF_Tensor*, 1> output_values{};

    // Run the session
    TF_SessionRun(session,
        run_options,
        input_ops.data(), input_values.data(), input_ops.size(),
        output_ops.data(), output_values.data(), output_ops.size(),
        nullptr, 0,
        nullptr,
        status);
    if (TF_GetCode(status) != TF_OK) {
        std::cout << TF_Message(status) << '\n';
    }

    // Check the results
    auto* output_tensor = static_cast<std::array<std::array<std::array<float, 3>, 128>, 128> *>(TF_TensorData(output_values[0]));
    std::vector<std::array<std::array<std::array<float, 3>, 128>, 128>> outputs{ output_tensor, output_tensor + inputs.size() };
    cv::Mat sr(hr.rows, hr.cols, CV_32FC3);         // super resolution
    float* sr_ptr = reinterpret_cast<float*>(sr.data);
    for (int i = 0; i < sr.rows; i++)
    {
        for (int j = 0; j < sr.cols; j++)
        {
            for (int c = 0; c < sr.channels(); c++)
            {
                sr_ptr[i * sr.cols * sr.channels() + j * sr.channels() + c] = outputs[0][i][j][c];
            }
        }
    }

    // Clear resources
    TF_DeleteTensor(input_values[0]);
    TF_DeleteTensor(output_values[0]);
    TF_DeleteBuffer(run_options);
    TF_DeleteSessionOptions(session_options);
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
}

void example_classification()
{
    /* https://int-i.github.io/cpp/2020-11-18/libtensorflow/ */
    // Show version information of the tensorflow
    std::cout << "TensorFlow Version: " << TF_Version() << std::endl;

    // Do initializations for tensorflow session
    auto* run_options = TF_NewBufferFromString("", 0);
    auto* session_options = TF_NewSessionOptions();
    auto* graph = TF_NewGraph();
    auto* status = TF_NewStatus();
    std::array<char const*, 1> tags{ "serve" };

    // Load the tensorflow model
    const std::string model_path = "../../classification/flower_classification";
    auto* session = TF_LoadSessionFromSavedModel(session_options, run_options,
        model_path.c_str(), tags.data(), tags.size(),
        graph, nullptr, status);
    if (TF_GetCode(status) != TF_OK) {
        std::cout << TF_Message(status) << '\n';
    }

    // Define network operations
    const std::string input_op_name = "serving_default_sequential_1_input";
    auto* input_op = TF_GraphOperationByName(graph, input_op_name.c_str());
    if (input_op == nullptr) {
        std::cout << "Failed to find graph operation\n";
    }
    const std::string output_op_name = "StatefulPartitionedCall";
    auto* output_op = TF_GraphOperationByName(graph, output_op_name.c_str());
    if (output_op == nullptr) {
        std::cout << "Failed to find graph operation\n";
    }
    std::array<TF_Output, 1> input_ops = { TF_Output{ input_op, 0 } };
    std::array<TF_Output, 1> output_ops = { TF_Output{ output_op, 0 } };

    // Load an input image
    cv::Mat daisy = cv::imread("../../classification/daisy0.jpg");          // class 0
    cv::Mat dandelion = cv::imread("../../classification/dandelion6.jpg");          // class 1
    cv::Mat roses = cv::imread("../../classification/roses7.jpg");          // class 2
    cv::Mat sunflowers = cv::imread("../../classification/sunflowers2.jpg");          // class 3
    cv::Mat tulips = cv::imread("../../classification/tulips5.jpg");          // class 4
    daisy.convertTo(daisy, CV_32F);
    dandelion.convertTo(dandelion, CV_32F);
    roses.convertTo(roses, CV_32F);
    sunflowers.convertTo(sunflowers, CV_32F);
    tulips.convertTo(tulips, CV_32F);

    // Set input data
    cv::Mat* img = &sunflowers;
    std::array<std::array<std::array<float, 3>, 180>, 180> x;
    float* img_ptr = reinterpret_cast<float*>(img->data);
    for (int i = 0; i < img->rows; i++)
    {
        for (int j = 0; j < img->cols; j++)
        {
            for (int c = 0; c < img->channels(); c++)
            {
                x[i][j][c] = img_ptr[i * img->cols * img->channels() + j * img->channels() + c];
            }
        }
    }
    std::vector<std::array<std::array<std::array<float, 3>, 180>, 180>> inputs{ x };
    std::array<int64_t, 4> const dims{ static_cast<int64_t>(inputs.size()), 180, 180, 3 };
    void* data = (void*)inputs.data();
    std::size_t const ndata = inputs.size() * 180 * 180 * 3 * TF_DataTypeSize(TF_FLOAT);
    auto const deallocator = [](void*, std::size_t, void*) {}; // unused deallocator because of RAII
    auto* input_tensor = TF_NewTensor(TF_FLOAT, dims.data(), dims.size(), data, ndata, deallocator, nullptr);
    std::array<TF_Tensor*, 1> input_values{ input_tensor };
    std::array<TF_Tensor*, 1> output_values{};

    // Run the session
    TF_SessionRun(session,
        run_options,
        input_ops.data(), input_values.data(), input_ops.size(),
        output_ops.data(), output_values.data(), output_ops.size(),
        nullptr, 0,
        nullptr,
        status);
    if (TF_GetCode(status) != TF_OK) {
        std::cout << TF_Message(status) << '\n';
    }

    // Check the results
    auto* output_tensor = static_cast<std::array<float, 5> *>(TF_TensorData(output_values[0]));
    std::vector<std::array<float, 5>> outputs{ output_tensor, output_tensor + inputs.size() };
    float max = outputs[0][0];
    int argmax = 0;
    for (int i = 0; i < 5; i++)
    {
        std::cout << outputs[0][i];         // classification result
        if (i == 4)
        {
            std::cout << std::endl;
        }
        else
        {
            std::cout << ",";
        }
        if (max < outputs[0][i])
        {
            max = outputs[0][i];
            argmax = i;
        }
    }
    
    // Show the results
    switch (argmax)
    {
    case 0: std::cout << "classification result : daisy" << std::endl; break;
    case 1: std::cout << "classification result : dandelion" << std::endl; break;
    case 2: std::cout << "classification result : roses" << std::endl; break;
    case 3: std::cout << "classification result : sunflowers" << std::endl; break;
    case 4: std::cout << "classification result : tulips" << std::endl; break;
    }

    // Clear resources
    TF_DeleteTensor(input_values[0]);
    TF_DeleteTensor(output_values[0]);
    TF_DeleteBuffer(run_options);
    TF_DeleteSessionOptions(session_options);
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
}

int main()
{
    example_classification();
}

// 프로그램 실행: <Ctrl+F5> 또는 [디버그] > [디버깅하지 않고 시작] 메뉴
// 프로그램 디버그: <F5> 키 또는 [디버그] > [디버깅 시작] 메뉴

// 시작을 위한 팁: 
//   1. [솔루션 탐색기] 창을 사용하여 파일을 추가/관리합니다.
//   2. [팀 탐색기] 창을 사용하여 소스 제어에 연결합니다.
//   3. [출력] 창을 사용하여 빌드 출력 및 기타 메시지를 확인합니다.
//   4. [오류 목록] 창을 사용하여 오류를 봅니다.
//   5. [프로젝트] > [새 항목 추가]로 이동하여 새 코드 파일을 만들거나, [프로젝트] > [기존 항목 추가]로 이동하여 기존 코드 파일을 프로젝트에 추가합니다.
//   6. 나중에 이 프로젝트를 다시 열려면 [파일] > [열기] > [프로젝트]로 이동하고 .sln 파일을 선택합니다.
