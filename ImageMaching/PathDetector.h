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
	�t�H���_���̑S�摜�̃p�X���擾��,���炩���ߌ��߂������K���Ɋ�Â��ĕ��ނ��A�O���[�v���Ƃɂ܂Ƃ߂ĕԂ�
	�����K��:����.���̕��̂��������_.�g���q(apple.view001.jpg)
	�摜���ꖇ�Ȃ�1��Ԃ�
	�摜������������A�O���[�v���ł�����0��Ԃ�
	*/
	int getPath(String^ folder, std::vector<std::string>&  paths)
	{


		/* step1: �w��t�H���_���̉摜path��file�ɕۑ� */
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
	System::String^ -> std::string �̕ϊ�
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