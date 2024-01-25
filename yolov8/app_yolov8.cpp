#include"../utils/yolo.h"
#include"yolov8.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>
#include "cpp_ai_utils.h"

namespace py = pybind11;

void setParameters(utils::InitParameter& initParameters)
{
	initParameters.class_names = utils::dataSets::coco80;
	//initParameters.class_names = utils::dataSets::voc20;
	initParameters.num_class = 80; // for coco
	//initParameters.num_class = 20; // for voc2012
	initParameters.batch_size = 8;
	initParameters.dst_h = 640;
	initParameters.dst_w = 640;
	initParameters.input_output_names = { "images",  "output0" };
	initParameters.conf_thresh = 0.25f;
	initParameters.iou_thresh = 0.45f;
	initParameters.save_path = "";
}

std::vector<std::vector<utils::Box>> task(YOLOV8& yolo, const utils::InitParameter& param, std::vector<cv::Mat>& imgsBatch, const int& delayTime, const int& batchi,
	const bool& isShow, const bool& isSave, const bool& isTrack, cpp_ai_utils::CppAiHelper& cppAiHelper, byte_track::BYTETracker& tracker, const std::string& queueName = "")
{
	utils::DeviceTimer d_t0; yolo.copy(imgsBatch);	      float t0 = d_t0.getUsedTime();
	utils::DeviceTimer d_t1; yolo.preprocess(imgsBatch);  float t1 = d_t1.getUsedTime();
	utils::DeviceTimer d_t2; yolo.infer();				  float t2 = d_t2.getUsedTime();
	utils::DeviceTimer d_t3; yolo.postprocess(imgsBatch); float t3 = d_t3.getUsedTime();
	/*
	sample::gLogInfo << 
		//"copy time = " << t0 / param.batch_size << "; "
		"preprocess time = " << t1 / param.batch_size << "; "
		"infer time = " << t2 / param.batch_size << "; "
		"postprocess time = " << t3 / param.batch_size << std::endl;
	*/

	std::vector<std::vector<utils::Box>> objectss;

	if (isTrack) {
		for (auto& frame : yolo.getObjectss()) {
			auto inputs = utils::convertBoxesToByteTrackObjects(frame);
			auto outputs = tracker.update(inputs);

			//sample::gLogInfo << "outputs:" << std::endl;
			//for (const auto& output : outputs) {
			//	auto& rect = output->getRect();
			//	sample::gLogInfo << cv::format("%f %f %f %f %d", rect.tl_x(), rect.tl_y(), rect.br_x(), rect.br_y(), output->getTrackId()) << std::endl;
			//}

			utils::setTrackIdToBoxes(frame, outputs);
			frame.erase(std::remove_if(frame.begin(), frame.end(), [](utils::Box box) { return box.track_id == -1; }), frame.end());
			objectss.push_back(frame);
		}
	} else {
		for (const auto& frame : yolo.getObjectss()) {
			std::vector<utils::Box> frameCopy;
			for (const auto& box : frame) {
				frameCopy.push_back(box);  // 假设 Box 有适当的复制构造函数
			}
			objectss.push_back(frameCopy);
		}
	}

	if(isShow)
		utils::show(objectss, param.class_names, delayTime, imgsBatch, cppAiHelper, queueName);
	if(isSave)
		utils::save(objectss, param.class_names, param.save_path, imgsBatch, param.batch_size, batchi, cppAiHelper);
	yolo.reset();
	return objectss;
}

std::vector<std::vector<utils::Box>> main_func(int argc, char** argv)
{
	std::vector<std::vector<utils::Box>> results;

	cv::CommandLineParser parser(argc, argv,
		{
			"{model|| tensorrt model file	   }"
			"{size|| image (h, w), eg: 640   }"
			"{batch_size|| batch size              }"
			"{video|| video's path			   }"
			"{img|| image's path			   }"
			"{cam_id|| camera's device id	   }"
			"{show|| if show the result	   }"
			"{savePath|| save path, can be ignore}"
			"{queueName|| camera jpg data queue   }"
			"{stopSignalKey|| stop camera signal key  }"
			"{logKey||log key}"
			"{doneKey||done key}"
			"{track||if track mode}"
			"{videoOutputPath||video output path}"
			"{videoProgressKey||video progress key}"
			"{videoOutputJsonPath||video output Json path}"
		});
	// parameters
	utils::InitParameter param;
	setParameters(param);
	// path
	std::string model_path = "../../data/yolov8/yolov8n.trt";
	std::string video_path = "../../data/people.mp4";
	std::string image_path = "../../data/bus.jpg";
	// camera' id
	int camera_id = 0;
	std::string queueName = "";
	std::string stopSignalKey = "";
	std::string logKey = "";
	std::string doneKey = "";
	std::string videoOutputPath = "";
	std::string videoProgressKey = "";
	std::string videoOutputJsonPath = "";

	// get input
	utils::InputStream source;
	source = utils::InputStream::IMAGE;
	//source = utils::InputStream::VIDEO;
	//source = utils::InputStream::CAMERA;

	// update params from command line parser
	int size = -1; // w or h
	int batch_size = 8;
	bool is_show = false;
	bool is_save = false;
	bool is_track = false;
	if (parser.has("size"))
	{
		size = parser.get<int>("size");
		sample::gLogInfo << "size = " << size << std::endl;
		param.dst_h = param.dst_w = size;
	}
	if (parser.has("queueName")) {
		queueName = parser.get<std::string>("queueName");
		sample::gLogInfo << "queueName = " << queueName << std::endl;
	}
	if (parser.has("stopSignalKey")) {
		stopSignalKey = parser.get<std::string>("stopSignalKey");
		sample::gLogInfo << "stopSignalKey = " << stopSignalKey << std::endl;
	}
	if (parser.has("model"))
	{
		model_path = parser.get<std::string>("model");
		sample::gLogInfo << "model_path = " << model_path << std::endl;
	}
	if (parser.has("batch_size"))
	{
		batch_size = parser.get<int>("batch_size");
		sample::gLogInfo << "batch_size = " << batch_size << std::endl;
		param.batch_size = batch_size;
	}
	if (parser.has("video"))
	{
		source = utils::InputStream::VIDEO;
		video_path = parser.get<std::string>("video");
		sample::gLogInfo << "video_path = " << video_path << std::endl;
	}
	if (parser.has("img"))
	{
		source = utils::InputStream::IMAGE;
		image_path = parser.get<std::string>("img");
		sample::gLogInfo << "image_path = " << image_path << std::endl;
	}
	if (parser.has("cam_id"))
	{
		source = utils::InputStream::CAMERA;
		camera_id = parser.get<int>("cam_id");
		sample::gLogInfo << "camera_id = " << camera_id << std::endl;
	}
	if (parser.has("show"))
	{
		is_show = true;
		sample::gLogInfo << "is_show = " << is_show << std::endl;
	}
	if (parser.has("savePath"))
	{
		is_save = true;
		param.save_path = parser.get<std::string>("savePath");
		sample::gLogInfo << "save_path = " << param.save_path << std::endl;
	}
	if (parser.has("logKey")) {
		logKey = parser.get<std::string>("logKey");
		sample::gLogInfo << "logKey = " << logKey << std::endl;
	}
	byte_track::BYTETracker tracker(30, 30);
	if (parser.has("track")) {
		is_track = true;
		sample::gLogInfo << "is_track = " << is_track << std::endl;
	}
	if (parser.has("doneKey")) {
		doneKey = parser.get<std::string>("doneKey");
		sample::gLogInfo << "doneKey = " << doneKey << std::endl;
	}
	if (parser.has("videoOutputPath")) {
		videoOutputPath = parser.get<std::string>("videoOutputPath");
		sample::gLogInfo << "videoOutputPath = " << videoOutputPath << std::endl;
	}
	if (parser.has("videoProgressKey")) {
		videoProgressKey = parser.get<std::string>("videoProgressKey");
		sample::gLogInfo << "videoProgressKey = " << videoProgressKey << std::endl;
	}
	if (parser.has("videoOutputJsonPath")) {
		videoOutputJsonPath = parser.get<std::string>("videoOutputJsonPath");
		sample::gLogInfo << "videoOutputJsonPath = " << videoOutputJsonPath << std::endl;
	}
	sample::gLogInfo << "is_save = " << is_save << std::endl;
	cpp_ai_utils::CppAiHelper cppAiHelper(logKey, queueName, stopSignalKey, 
		videoOutputPath, videoProgressKey, videoOutputJsonPath);

	int total_batches = 0;
	int delay_time = 1;
	cv::VideoCapture capture;
	if (!setInputStream(source, image_path, video_path, camera_id,
		capture, total_batches, delay_time, param))
	{
		sample::gLogError << "read the input data errors!" << std::endl;
		cppAiHelper.push_log_to_redis(u8"读取输入数据出错");
		sample::gLogInfo << "push log to redis done." << std::endl;
		return results;
	}
	if (source == utils::InputStream::VIDEO || source == utils::InputStream::CAMERA) {
		cppAiHelper.init_video_writer(capture);
	}

	YOLOV8 yolo(param);

	// read model
	std::vector<unsigned char> trt_file = utils::loadModel(model_path);
	if (trt_file.empty())
	{
		sample::gLogError << "trt_file is empty!" << std::endl;
		cppAiHelper.push_log_to_redis(u8"读取输入数据出错");
		return results;
	}
	// init model
	if (!yolo.init(trt_file))
	{
		sample::gLogError << "initEngine() ocur errors!" << std::endl;
		cppAiHelper.push_log_to_redis(u8"读取输入数据出错");
		return results;
	}
	yolo.check();
	cv::Mat frame;
	std::vector<cv::Mat> imgs_batch;
	imgs_batch.reserve(param.batch_size);
	sample::gLogInfo << imgs_batch.capacity() << std::endl;
	int batchi = 0;
	bool stopFlag = false;
	while (capture.isOpened())
	{
		bool shouldStop = cppAiHelper.should_stop_camera();
		if (shouldStop) {
			break;
		}
		
		if (batchi >= total_batches && source != utils::InputStream::CAMERA)
		{
			break;
		}
		if (imgs_batch.size() < param.batch_size) // get input
		{
			if (source != utils::InputStream::IMAGE)
			{
				capture.read(frame);
			}
			else
			{
				frame = cv::imread(image_path);
			}

			if (frame.empty())
			{
				sample::gLogWarning << "no more video or camera frame" << std::endl;
				auto r = task(yolo, param, imgs_batch, delay_time, batchi, is_show, is_save, is_track, cppAiHelper, tracker, queueName);
				if (source == utils::InputStream::IMAGE) {
					results.insert(results.end(), r.begin(), r.end());
				}
				imgs_batch.clear();
				batchi++;
				break;
			}
			else
			{
				imgs_batch.emplace_back(frame.clone());
			}
		}
		else
		{
			auto r = task(yolo, param, imgs_batch, delay_time, batchi, is_show, is_save, is_track, cppAiHelper, tracker, queueName);
			if (source == utils::InputStream::IMAGE) {
				results.insert(results.end(), r.begin(), r.end());
			}
			imgs_batch.clear();
			batchi++;
		}
	}

	return results;
}

std::vector<std::vector<utils::Box>> main_func_wrapper(const std::vector<std::string>& strings) {
	int argc = strings.size();
	std::vector<char*> cstrings;
	cstrings.reserve(strings.size());
	for (size_t i = 0; i < strings.size(); ++i) {
		cstrings.push_back(const_cast<char*>(strings[i].c_str()));
	}
	/*std::cout << "argc = " << argc << std::endl;
	for (size_t i = 0; i < cstrings.size(); ++i)
		std::cout << "cs[" << i << "] = " << cstrings[i] << std::endl;*/
	auto r = main_func(argc, &cstrings[0]);
	return r;
}

PYBIND11_MODULE(app_yolo, m) {
	m.doc() = "yolov8 pyd"; // optional module docstring

	py::class_<utils::Box>(m, "Box")
		.def_readwrite("left", &utils::Box::left)
		.def_readwrite("top", &utils::Box::top)
		.def_readwrite("right", &utils::Box::right)
		.def_readwrite("bottom", &utils::Box::bottom)
		.def_readwrite("confidence", &utils::Box::confidence)
		.def_readwrite("track_id", &utils::Box::track_id)
		.def_readwrite("label", &utils::Box::label);

	m.def("main_func_wrapper", &main_func_wrapper, "main func");
}
