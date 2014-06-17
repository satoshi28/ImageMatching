#ifndef PATH_DETECTOR
#define PATH_DETECTOR

#include <cliext/vector>

#include <vector>
#include <cstdlib>

using namespace System;
using namespace System::IO;

class PathDetector
{
public:
	PathDetector();
	~PathDetector();

public:
	/*
	フォルダ内の全画像のパスを取得し,あらかじめ決めた命名規則に基づいて分類し、グループごとにまとめて返す
	命名規則:物体.その物体を見た視点.拡張子(apple.view001.jpg)
	画像が一枚なら1を返す
	画像が複数枚あり、グループ化できたら0を返す
	*/
	int getPath(String^ folder, std::vector<std::string>&  paths)
	{


		/* step1: 指定フォルダ下の画像pathをfileに保存 */
		array<String^>^ file = Directory::GetFiles( folder );
	
		std::string path;
		for(int i = 0; i < file->Length; i++)
		{
			MarshalString(file[i], path);
			paths.push_back(path);

			path.clear();
		}

		
		return 0;
	}

private:
	/*
	System::String^ -> std::string の変換
	*/
	void MarshalString ( String ^ s, std::string& os ) 
	{
	using namespace Runtime::InteropServices;
	const char* chars = 
	   (const char*)(Marshal::StringToHGlobalAnsi(s)).ToPointer();
	os = chars;
	Marshal::FreeHGlobal(IntPtr((void*)chars));
	}

};

PathDetector::PathDetector()
{
}

PathDetector::~PathDetector()
{
}

#endif