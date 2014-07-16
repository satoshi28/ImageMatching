#include <iostream>
#include <string>
#include <fstream>
#include <functional>
#include <opencv2\opencv.hpp>
#include <opencv2\nonfree\nonfree.hpp>

#include "PathDetector.h"

using namespace std;
using namespace cv;

float minRatio = 0.8f;

bool readImages(std::vector<std::string> filenames, std::vector<cv::Mat>& images);
void match(std::vector<cv::KeyPoint> queryKeypoints, cv::Mat queryDescriptors, std::vector<cv::KeyPoint> trainKeypoints,cv::Ptr<cv::DescriptorMatcher>& m_matcher,
				std::vector<cv::DMatch>& matches);
bool geometricConsistencyCheck(std::vector<cv::KeyPoint> queryKeypoints, std::vector<cv::KeyPoint> trainKeypoints, std::vector<cv::DMatch>& match);
int getResult(std::vector<std::string> filelist, cv::Mat& resultImage);

std::vector<cv::vector<cv::KeyPoint>> queryKeypoints;
std::vector<cv::vector<cv::KeyPoint>> trainKeypoints;

int main(int argc, char *argv[])
{
	System::String^ IMAGE_DIR = "C:\\Users\\satoshi\\Documents\\Image\\ZuBuD\\query\\";	// �摜���ۑ�����Ă���t�H���_
	System::String^ DATABASE_IMG_DIR = "C:\\Users\\satoshi\\Documents\\Image\\ZuBuD\\database\\";	// �摜���ۑ�����Ă���t�H���_

	std::ofstream txtFile("matchingReslut_BRISK.txt");

	//1�}�C�N���b�������tick��
	double ticFrequency = cvGetTickFrequency();
	double processTime;
	int startTic;
	int stopTic;

	PathDetector path;

	// �t�H���_�̉摜���𑖍�
	std::vector<std::string> backfilelist;	//�摜�̐�΃p�X
	std::vector<cv::Mat> queryImages;

	//�f�[�^�x�[�X�摜�̓ǂݍ���
	std::vector<std::string> databaseFilelist;	//�摜�̐�΃p�X
	std::vector<cv::Mat> databaseImages;

	path.getPath(IMAGE_DIR, backfilelist);
	path.getPath(DATABASE_IMG_DIR, databaseFilelist);

	// (1)load Color Image
	readImages(backfilelist, queryImages);
	readImages(databaseFilelist, databaseImages);

	
	SurfFeatureDetector detector(400);
	//SurfDescriptorExtractor extractor;
	//FastFeatureDetector detector(30);
	//cv::Ptr<cv::FeatureDetector>     detector  = new cv::OrbFeatureDetector(1500,2.0f); 
    //cv::Ptr<cv::DescriptorExtractor> extractor = new cv::OrbDescriptorExtractor;
	//cv::Ptr<cv::FeatureDetector>     detector  = cv::FeatureDetector::create("PyramidSTAR"); 
    cv::Ptr<cv::DescriptorExtractor> extractor = new cv::BRISK;
	///cv::Ptr<cv::FeatureDetector>     detector  = new cv::SURF(400); 
    //cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SURF;

	//OrbDescriptorExtractor extractor; //ORB�����ʒ��o�@

	std::vector< cv::Ptr<cv::DescriptorMatcher> > matchers;

	std::vector<cv::Mat> queryDescriptors;

	int descSizeRow = 0;
	/* ���Ԍv���X�^�[�g */
	int totalTime = 0;
	int totalKeyPoint = 0;
	startTic = cvGetTickCount();

	for(int i =0; i < queryImages.size();i++)
	{
		cv::Mat descriptors;
		cv::vector<cv::KeyPoint> keypoints;

		startTic = cvGetTickCount();
		detector.detect(queryImages[i],keypoints);
		stopTic = cvGetTickCount();

		totalTime += stopTic - startTic;
		totalKeyPoint += keypoints.size();

		extractor->compute(queryImages[i],keypoints,descriptors);

		queryKeypoints.push_back(keypoints);
		queryDescriptors.push_back(descriptors);
	}
	  std::cout << "average_detect_time:"<<totalTime/ ticFrequency/queryImages.size()/1000 << "[ms]"<< std::endl;
	  std::cout << "keypoint/time:"<<totalTime/ ticFrequency/totalKeyPoint << "[us]"<< std::endl;
	
	std::cout << descSizeRow << std::endl;
	descSizeRow =0;

	for(int i =0; i < databaseImages.size();i++)
	{
		std::vector<cv::Mat> descriptors(1);
		cv::vector<cv::KeyPoint> keypoints;
		detector.detect(databaseImages[i],keypoints);
		extractor->compute(databaseImages[i],keypoints,descriptors[0]);

		descSizeRow += descriptors[0].rows;

		cv::Ptr<cv::DescriptorMatcher>   matcher   = new cv::BFMatcher(NORM_HAMMING,false);
		matcher->add(descriptors);
		matcher->train();

		trainKeypoints.push_back(keypoints);
		matchers.push_back(matcher);
	}

	std::cout << descSizeRow << std::endl;

	

	for(int i =0; i< queryDescriptors.size(); i++)
	{
		std::vector< std::pair<int, int> > imageRankingList(databaseImages.size());	//�e�摜�̃����L���O(rank, index)

		for(int j = 0; j < databaseImages.size();j++)
		{
			std::pair<int, int> list;
			std::vector<cv::DMatch> matches;

			//�}�b�`���O
			match(queryKeypoints[i], queryDescriptors[i],trainKeypoints[j],matchers[j], matches);
			

			list.first = matches.size();
			list.second = j;

			imageRankingList.push_back(list);
		}

		//�摜�̃����L���O�Ɋ�Â��č~���ɕ��ёւ�
		std::sort(imageRankingList.begin(), imageRankingList.end(),std::greater<std::pair<int, int>>() );
		
		cv::Mat matchingResult;
		std::vector<std::string> matchingList;

		matchingList.push_back(backfilelist[i]);		//query�摜�̃p�X��ۑ�
		//3�ʂ܂Ō��ʂ��o��
		for(int i = 0; i < 3; i++)
			matchingList.push_back(databaseFilelist[imageRankingList[i].second]);	//�������ʂ̕ۑ�

		//���ʂ��e�L�X�g�t�@�C���ɏo��
		txtFile << matchingList[0] << "	" << matchingList[1] << std::endl;
		//���ʂ��摜�ŏo��
		getResult(matchingList, matchingResult);
		
		static int count = 0;
		std::stringstream ss;
		ss << count;
		std::string result = "result";
		result +=  ss.str();
		result += ".jpg";
		cv::imwrite(result,matchingResult);
		count++;

		std::cout << count << std::endl;
	}
	
  //cv::imwrite("BRISK.jpg",colorImage);

  std::cout << (stopTic-startTic)/ ticFrequency << "�}�C�N���b"<< std::endl;

  cv::waitKey(0);
}

void match(std::vector<cv::KeyPoint> queryKeypoints, cv::Mat queryDescriptors, std::vector<cv::KeyPoint> trainKeypoints,cv::Ptr<cv::DescriptorMatcher>& m_matcher,
				std::vector<cv::DMatch>& matches)
{
	matches.clear();

	//�ŋߖT�_�̒T��

	//knn�}�b�`���O
	std::vector< std::vector<cv::DMatch>>  knnMatches;

	// query��matcher�ɕۑ�����Ă�������ʂ�knn�\���̂�p���čŋߖT�_����������.
	m_matcher->knnMatch(queryDescriptors, knnMatches, 2);

	//
	std::vector<cv::DMatch> correctMatches;

	//ratio test
	for(int j = 0; j < knnMatches.size(); j++)
	{
		if(knnMatches[j].empty() == false)
		{
			const cv::DMatch& bestMatch = knnMatches[j][0];
			const cv::DMatch& betterMatch = knnMatches[j][1];

			float distanceRatio = bestMatch.distance / betterMatch.distance;

			//�����̔䂪1.5�ȉ��̓��������ۑ�
			if(distanceRatio < minRatio)
			{
				matches.push_back(bestMatch);
			}
		}
	}

	//�􉽊w�I�������`�F�b�N
	bool passFlag = geometricConsistencyCheck(queryKeypoints, trainKeypoints, matches);


	
}

bool readImages(std::vector<std::string> filenames, std::vector<cv::Mat>& images)
{
	for(int i = 0; i < filenames.size(); i++)
	{

		cv::Mat image;
		std::string a = filenames[i];
		
		//std::cout << a << std::endl;

		image = cv::imread(a,1);			//�摜�̓ǂݍ���
		if (image.empty())
		{
			std::cout << "Input image cannot be read" << std::endl;
			return false;
		}

		Mat grayImage;
		cvtColor(image, grayImage, CV_BGR2GRAY);

		//�摜��ǉ�
		images.push_back(grayImage);

	}
}

bool geometricConsistencyCheck(std::vector<cv::KeyPoint> queryKeypoints, std::vector<cv::KeyPoint> trainKeypoints, std::vector<cv::DMatch>& match)
{
	if(match.size() < 8)
	{
		match.clear();
		return false;

	}
	std::vector<cv::Point2f>  queryPoints, trainPoints; 
	for(int i = 0; i < match.size(); i++)
	{
		queryPoints.push_back(queryKeypoints[match[i].queryIdx].pt);
		trainPoints.push_back(trainKeypoints[match[i].trainIdx].pt);
	}

	//�􉽊w�I�������`�F�b�N
	std::vector<unsigned char> inliersMask(queryPoints.size() );

	//�􉽊w�I�������`�F�b�N�ɂ���ē�����l�𒊏o
	cv::findHomography( queryPoints, trainPoints, CV_FM_RANSAC, 10, inliersMask);

	std::vector<cv::DMatch> inliers;
	for(size_t i =0 ; i < inliersMask.size(); i++)
	{
		if(inliersMask[i])
			inliers.push_back(match[i]);
	}

	match.swap(inliers);
	return true;
}

int getResult(std::vector<std::string> filelist, cv::Mat& resultImage)
{
	int total_width =0;
	int height = 0;
	int width = 0;
	std::vector<cv::Mat> images;
	for(int i = 0; i < filelist.size(); i++)
	{

		cv::Mat image;
		cv::Mat dstImage;
		image = cv::imread(filelist[i],1);			//�摜�̓ǂݍ���
		if (image.empty())
		{
			std::cout << "Input image cannot be read" << std::endl;
			return false;
		}

		if(i == 0)
		{
			height = image.rows;
			width = image.cols;
			dstImage = image;
		}else
		{
			float rx = (float)height / (float)image.rows;
			float ry = (float)width / (float)image.cols;
			cv::resize(image, dstImage, cv::Size(), rx, ry);
			 
			
		}

		total_width += dstImage.cols;
		//�摜��ǉ�
		images.push_back(dstImage);
	}

	cv::Mat combinedImage(cv::Size(total_width, height), CV_8UC3);

	std::vector<cv::Mat>::iterator it  = images.begin(), it_end = images.end();
	cv::Rect roi_rect;
	int cnt = 0;
	for (; it != it_end; ++it) {
		roi_rect.width = it->cols;
		roi_rect.height = it->rows;
 
		cv::Mat roi(combinedImage, roi_rect);
		it->copyTo(roi);
		roi_rect.x += it->cols;
 
		cnt++;
	}
	resultImage = combinedImage;
}
