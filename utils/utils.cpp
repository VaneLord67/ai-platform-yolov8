#include"../utils/utils.h"
#include <cpp_redis/cpp_redis>

void utils::saveBinaryFile(float* vec, size_t len, const std::string& file)
{
	std::ofstream  out(file, std::ios::out | std::ios::binary);
	if (!out.is_open())
		return;
	out.write((const char*)vec, sizeof(float) * len);
	out.close();
}

std::vector<uint8_t> utils::readBinaryFile(const std::string& file) 
{

	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open())
		return {};

	in.seekg(0, std::ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0) {
		in.seekg(0, std::ios::beg);
		data.resize(length);

		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}


std::vector<unsigned char> utils::loadModel(const std::string& file)
{
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
    {
        return {};
    }
    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0)
    {
        in.seekg(0, std::ios::beg);
        data.resize(length);
        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

std::string utils::getSystemTimeStr()
{
	return std::to_string(std::rand()); 
}

bool utils::setInputStream(const utils::InputStream& source, const std::string& imagePath, const std::string& videoPath, const int& cameraID,
	cv::VideoCapture& capture, int& totalBatches, int& delayTime, utils::InitParameter& param)
{
	int total_frames = 0;
	std::string img_format;
	switch (source)
	{
	case utils::InputStream::IMAGE:
		img_format = imagePath.substr(imagePath.size()-4, 4);
		if (img_format == ".png" || img_format == ".PNG")
		{
			sample::gLogWarning << "+-----------------------------------------------------------+" << std::endl;
			sample::gLogWarning << "| If you use PNG format pictures, the file name must be eg: |" << std::endl;
			sample::gLogWarning << "| demo0.png, demo1.png, demo2.png ......, but not demo.png. |" << std::endl;
			sample::gLogWarning << "| The above rules are determined by OpenCV.					|" << std::endl;
			sample::gLogWarning << "+-----------------------------------------------------------+" << std::endl;
		}
		capture.open(imagePath); //cv::CAP_IMAGES : !< OpenCV Image Sequence (e.g. img_%02d.jpg)
		param.batch_size = 1;
		total_frames = 1;
		totalBatches = 1;
		delayTime = 0;
		break;
	case utils::InputStream::VIDEO:
		capture.open(videoPath);
		total_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
		totalBatches = (total_frames % param.batch_size == 0) ?
			(total_frames / param.batch_size) : (total_frames / param.batch_size + 1);
		break;
	case utils::InputStream::CAMERA:
		capture.open(cameraID);
		total_frames = INT_MAX;
		totalBatches = INT_MAX;
		break;
	default:
		break;
	}
	param.src_h = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	param.src_w = capture.get(cv::CAP_PROP_FRAME_WIDTH);

	return capture.isOpened();
}

void utils::setRenderWindow(InitParameter& param)
{
	if (!param.is_show)
		return;
	int max_w = 960;
	int max_h = 540;
	float scale_h = (float)param.src_h / max_h;
	float scale_w = (float)param.src_w / max_w;
	if (scale_h > 1.f && scale_w > 1.f)
	{
		float scale = scale_h < scale_w ? scale_h : scale_w;
		cv::namedWindow(param.winname, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);  // for Linux
		cv::resizeWindow(param.winname, int(param.src_w / scale), int(param.src_h / scale));
		param.char_width = 16;
		param.det_info_render_width = 18;
		param.font_scale = 0.9;
	}
	else
	{
		cv::namedWindow(param.winname);
	}
}

std::string utils::getTimeStamp()
{
	std::chrono::nanoseconds t = std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::system_clock::now().time_since_epoch());
	return std::to_string(t.count());
}

byte_track::Object utils::convertBoxToByteTrackObject(const utils::Box& box) {
	byte_track::Rect<float> rect(box.left, box.top, box.right - box.left, box.bottom - box.top);
	byte_track::Object obj(rect, box.label, box.confidence);
	return obj;
}

std::vector<byte_track::Object> utils::convertBoxesToByteTrackObjects(const std::vector<utils::Box> boxes) {
	std::vector<byte_track::Object> objects;
	for (const auto& box : boxes) {
		objects.push_back(utils::convertBoxToByteTrackObject(box));
	}
	return objects;
}

bool floatEqual(float a, float b) {
	return std::abs(a - b) < 1;
}

void utils::setTrackIdToBoxes(std::vector<utils::Box>& boxes, const std::vector<byte_track::BYTETracker::STrackPtr> outputs) {
	//sample::gLogInfo << std::endl;

	/*sample::gLogInfo << "boxes:" << std::endl;
	for (const auto& box : boxes) {
		sample::gLogInfo << cv::format("%f %f %f %f", box.left, box.top, box.right, box.bottom) << std::endl;
	}

	sample::gLogInfo << "outputs:" << std::endl;
	for (const auto& output : outputs) {
		auto& rect = output->getRect();
		sample::gLogInfo << cv::format("%f %f %f %f %d", rect.tl_x(), rect.tl_y(), rect.br_x(), rect.br_y(), output->getTrackId()) << std::endl;
	}*/

	for (auto& box : boxes) {
		for (const auto& output : outputs) {
			auto& rect = output->getRect();
			if (floatEqual(box.left, rect.tl_x()) && floatEqual(box.top, rect.tl_y()) && 
				floatEqual(box.right, rect.br_x()) && floatEqual(box.bottom, rect.br_y())) {
				box.track_id = output->getTrackId();
				// sample::gLogInfo << "set track_id" << std::endl;
				break;
			}
		}
	}
}

void utils::show(const std::vector<std::vector<utils::Box>>& objectss, const std::vector<std::string>& classNames,
	const int& cvDelayTime, std::vector<cv::Mat>& imgsBatch, cpp_ai_utils::CppAiHelper& cppAiHelper, const std::string& queueName)
{
	std::string windows_title = "image";
	if(!imgsBatch[0].empty() && queueName == "")
	{
		cv::namedWindow(windows_title, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);  // allow window resize(Linux)

		int max_w = 960;
		int max_h = 540;
		if (imgsBatch[0].rows > max_h || imgsBatch[0].cols > max_w)
		{
			cv::resizeWindow(windows_title, max_w, imgsBatch[0].rows * max_w / imgsBatch[0].cols );
		}
	}
	
	// vis
	cv::Scalar color = cv::Scalar(0, 255, 0);
	cv::Point bbox_points[1][4];
	const cv::Point* bbox_point0[1] = { bbox_points[0] };
	int num_points[] = { 4 };
	for (size_t bi = 0; bi < imgsBatch.size(); bi++)
	{
		if (!objectss.empty())
		{
			for (auto& box : objectss[bi])
			{
				if (classNames.size() == 91) // coco91
				{
					color = Colors::color91[box.label];
				}
				if (classNames.size() == 80) // coco80
				{
					color = Colors::color80[box.label];
				}
				if (classNames.size() == 20) // voc20
				{
					color = Colors::color80[box.label];
				}
				cv::rectangle(imgsBatch[bi], cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 2, cv::LINE_AA);
				cv::String det_info = classNames[box.label] + " " + cv::format("%.4f", box.confidence);
				if (box.track_id > 0) {
					det_info += cv::format(" %d", box.track_id);
				}
				bbox_points[0][0] = cv::Point(box.left, box.top);
				bbox_points[0][1] = cv::Point(box.left + det_info.size() * 11, box.top);
				bbox_points[0][2] = cv::Point(box.left + det_info.size() * 11, box.top - 15);
				bbox_points[0][3] = cv::Point(box.left, box.top - 15);
				cv::fillPoly(imgsBatch[bi], bbox_point0, num_points, 1, color);
				cv::putText(imgsBatch[bi], det_info, bbox_points[0][0], cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

				if (!box.land_marks.empty()) // for facial landmarks
				{
					for (auto& pt:box.land_marks)
					{
						cv::circle(imgsBatch[bi], pt, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA, 0);
					}
				}
			}
		}

		// camera_stream process
		if (queueName != "") {
			cppAiHelper.write_frame_to_video(imgsBatch[bi]);
			cppAiHelper.push_frame_to_redis(imgsBatch[bi]);

			std::stringstream boxJson;
			boxJson << "[";
			for (auto& box : objectss[bi]) {
				std::stringstream ss;
				ss << "{"
					<< "\"left\":" << box.left << ","
					<< "\"top\":" << box.top << ","
					<< "\"bottom\":" << box.bottom << ","
					<< "\"right\":" << box.right << ","
					<< "\"confidence\":" << box.confidence << ","
					<< "\"track_id\":" << box.track_id << ","
					<< "\"class_name\":\"" << classNames[box.label] << "\","
					<< "\"label\":" << box.label << ""
					<< "}";
				if (&box != &objectss[bi].back()) {
					ss << ",";
				}
				boxJson << ss.str();
			}
			boxJson << "]";

			cppAiHelper.push_str_to_redis(boxJson.str());
			cppAiHelper.write_json_to_file(boxJson.str());

		}

		if (queueName == "") {
			cv::imshow(windows_title, imgsBatch[bi]);
			cv::waitKey(cvDelayTime);
		}
	}
}

void utils::save(const std::vector<std::vector<Box>>& objectss, const std::vector<std::string>& classNames,
	const std::string& savePath, std::vector<cv::Mat>& imgsBatch, const int& batchSize, const int& batchi, 
	cpp_ai_utils::CppAiHelper& cppAiHelper)
{
	cv::Scalar color = cv::Scalar(0, 255, 0);
	cv::Point bbox_points[1][4];
	const cv::Point* bbox_point0[1] = { bbox_points[0] };
	int num_points[] = { 4 };
	for (size_t bi = 0; bi < imgsBatch.size(); bi++)
	{
		if (!objectss.empty())
		{
			for (auto& box : objectss[bi])
			{
				if (classNames.size() == 91) // coco91
				{
					color = Colors::color91[box.label];
				}
				if (classNames.size() == 80) // coco80
				{
					color = Colors::color80[box.label];
				}
				if (classNames.size() == 20) // voc20
				{
					color = Colors::color20[box.label];
				}
				cv::rectangle(imgsBatch[bi], cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 2, cv::LINE_AA);
				cv::String det_info = classNames[box.label] + " " + cv::format("%.4f", box.confidence);
				if (box.track_id > 0) {
					det_info += cv::format(" %d", box.track_id);
				}
				bbox_points[0][0] = cv::Point(box.left, box.top);
				bbox_points[0][1] = cv::Point(box.left + det_info.size() * 11, box.top);
				bbox_points[0][2] = cv::Point(box.left + det_info.size() * 11, box.top - 15);
				bbox_points[0][3] = cv::Point(box.left, box.top - 15);
				cv::fillPoly(imgsBatch[bi], bbox_point0, num_points, 1, color);
				cv::putText(imgsBatch[bi], det_info, bbox_points[0][0], cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
				
				if (!box.land_marks.empty())
				{
					for (auto& pt : box.land_marks)
					{
						cv::circle(imgsBatch[bi], pt, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA, 0);
					}
				}
			}
		}
		
		int imgi = batchi * batchSize + bi;
		
		if (cppAiHelper.get_source_type() != cpp_ai_utils::SourceType::VIDEO) {
			cv::imwrite(savePath + "_" + std::to_string(imgi) + ".jpg", imgsBatch[bi]);
			cv::waitKey(1); // waitting for writting imgs 
		} else {
			cppAiHelper.write_frame_to_video(imgsBatch[bi]);

			std::stringstream boxJson;
			boxJson << "[";
			for (auto& box : objectss[bi]) {
				std::stringstream ss;
				ss << "{"
					<< "\"left\":" << box.left << ","
					<< "\"top\":" << box.top << ","
					<< "\"bottom\":" << box.bottom << ","
					<< "\"right\":" << box.right << ","
					<< "\"confidence\":" << box.confidence << ","
					<< "\"track_id\":" << box.track_id << ","
					<< "\"class_name\":\"" << classNames[box.label] << "\","
					<< "\"label\":" << box.label << ""
					<< "}";
				if (&box != &objectss[bi].back()) {
					ss << ",";
				}
				boxJson << ss.str();
			}
			boxJson << "]";
			cppAiHelper.write_json_to_file(boxJson.str());
		}
		
	}
}

utils::HostTimer::HostTimer()
{
    t1 = std::chrono::steady_clock::now();
}

float utils::HostTimer::getUsedTime()
{
    t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    return(1000 * time_used.count()); // ms
}

utils::HostTimer::~HostTimer()
{
}

utils::DeviceTimer::DeviceTimer()
{
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
}

float utils::DeviceTimer::getUsedTime()
{
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float total_time;
	cudaEventElapsedTime(&total_time, start, end);
	return total_time;
}

utils::DeviceTimer::DeviceTimer(cudaStream_t stream)
{
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, stream);
}

float utils::DeviceTimer::getUsedTime(cudaStream_t stream)
{
	cudaEventRecord(end, stream);
	cudaEventSynchronize(end);
	float total_time;
	cudaEventElapsedTime(&total_time, start, end);
	return total_time;
}

utils::DeviceTimer::~DeviceTimer()
{
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}