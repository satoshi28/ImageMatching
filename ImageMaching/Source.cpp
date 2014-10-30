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
void match(cv::Mat queryDescriptors, cv::Ptr<cv::DescriptorMatcher>& m_matcher,
				std::vector<cv::DMatch>& matches);
int getResult(std::vector<std::string> filelist, cv::Mat& resultImage);


int main(int argc, char *argv[])
{
	System::String^ IMAGE_DIR = "C:\\Users\\satoshi\\Documents\\Image\\ZuBuD\\query";	// 画像が保存されているフォルダ
	System::String^ DATABASE_IMG_DIR = "C:\\Users\\satoshi\\Documents\\Image\\ZuBuD\\databaseImage";	// 画像が保存されているフォルダ

	std::ofstream txtFile("matchingReslut_SURF+BRISK.txt");

	//1マイクロ秒あたりのtick数
	double ticFrequency = cvGetTickFrequency();
	double processTime;
	int startTic;
	int stopTic;

	PathDetector path;

	// フォルダの画像名を走査
	std::vector<std::string> backfilelist;	//画像の絶対パス
	std::vector<cv::Mat> queryImages;

	//データベース画像の読み込み
	std::vector<std::string> databaseFilelist;	//画像の絶対パス
	std::vector<cv::Mat> databaseImages;

	path.getPath(IMAGE_DIR, backfilelist);
	path.getPath(DATABASE_IMG_DIR, databaseFilelist);

	// (1)load Color Image
	readImages(backfilelist, queryImages);
	readImages(databaseFilelist, databaseImages);

	
	SurfFeatureDetector detector(400);
    cv::Ptr<cv::DescriptorExtractor> extractor = new cv::BRISK;


	//OrbDescriptorExtractor extractor; //ORB特徴量抽出機

	std::vector<cv::Mat> queryDescriptors;

	int descSizeRow =0;

	/* 時間計測スタート */
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

		queryDescriptors.push_back(descriptors);
	}
	  std::cout << "average_detect_time:"<<totalTime/ ticFrequency/queryImages.size()/1000 << "[ms]"<< std::endl;
	  std::cout << "keypoint/time:"<<totalTime/ ticFrequency/totalKeyPoint << "[us]"<< std::endl;
	

	std::cout << descSizeRow << std::endl;
	descSizeRow = 0;

	std::vector<cv::Mat> trainDescriptors;
	for(int i =0; i < databaseImages.size();i++)
	{
		cv::Mat descriptors;
		cv::vector<cv::KeyPoint> keypoints;
		detector.detect(databaseImages[i],keypoints);
		extractor->compute(databaseImages[i],keypoints,descriptors);

		descSizeRow += descriptors.rows;

		trainDescriptors.push_back(descriptors);

	}
	printf("%i",descSizeRow);
	
	cv::Ptr<cv::flann::IndexParams>  index_params = new cv::flann::LshIndexParams(6,12,1);
	cv::Ptr<cv::flann::SearchParams> search_params = new cv::flann::SearchParams(50);
	cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(index_params, search_params);

	matcher.add(trainDescriptors);
	matcher.train();

	/*

	cv::Ptr<cv::DescriptorMatcher>   matcher   = cv::DescriptorMatcher::create("BruteForce-Hamming");
	matcher->add(trainDescriptors);
	matcher->train();
	*/
	std::cout << descSizeRow << std::endl;
	
	//マッチングしたペア
	std::vector<cv::DMatch> matches;

	for(int i =0; i< queryDescriptors.size(); i++)
	{
		std::vector< std::pair<int, int> > imageRankingList(databaseImages.size());	//各画像のランキング(rank, index)

		//マッチング
		//match(queryDescriptors[i], matcher, matches);
		matcher.match(queryDescriptors[i], matches);
		//初期化
		for(int i = 0; i < databaseImages.size(); i++)
		{
			imageRankingList[i].first = 0;
			imageRankingList[i].second = i;
		}

		//評価
		int num;
		for(int j = 0; j < matches.size(); j++)
		{
			num = matches[j].imgIdx;
			imageRankingList[num].first += 1;
		}

		//画像のランキングに基づいて降順に並び替え
		std::sort(imageRankingList.begin(), imageRankingList.end(),std::greater<std::pair<int, int>>() );
		
		cv::Mat matchingResult;
		std::vector<std::string> matchingList;

		matchingList.push_back(backfilelist[i]);		//query画像のパスを保存
		//3位まで結果を出す
		for(int i = 0; i < 3; i++)
			matchingList.push_back(databaseFilelist[imageRankingList[i].second]);	//検索結果の保存

		//結果をテキストファイルに出力
		txtFile << matchingList[0] << "	" << matchingList[1] << std::endl;
		//結果を画像で出力
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

  std::cout << (stopTic-startTic)/ ticFrequency << "マイクロ秒"<< std::endl;

  cv::waitKey(0);
}

void match(cv::Mat queryDescriptors, cv::Ptr<cv::DescriptorMatcher>& m_matcher,
				std::vector<cv::DMatch>& matches)
{
	matches.clear();

	//最近傍点の探索

	//knnマッチング
	std::vector< std::vector<cv::DMatch>>  knnMatches;

	// queryとmatcherに保存されている特徴量をknn構造体を用いて最近傍点を検索する.
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

			//距離の比が1.5以下の特徴だけ保存
			if(distanceRatio < minRatio)
			{
				matches.push_back(bestMatch);
			}
		}
	}
	
}

bool readImages(std::vector<std::string> filenames, std::vector<cv::Mat>& images)
{
	for(int i = 0; i < filenames.size(); i++)
	{

		cv::Mat image;
		std::string a = filenames[i];
		
		//std::cout << a << std::endl;

		image = cv::imread(a,1);			//画像の読み込み
		if (image.empty())
		{
			std::cout << "Input image cannot be read" << std::endl;
			return false;
		}

		Mat grayImage;
		cvtColor(image, grayImage, CV_BGR2GRAY);

		//画像を追加
		images.push_back(grayImage);

	}
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
		image = cv::imread(filelist[i],1);			//画像の読み込み
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
		//画像を追加
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
