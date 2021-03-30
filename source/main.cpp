#include <opencv2\opencv.hpp>
#include <iostream>
#include "fun.h"

using namespace cv;
using namespace std;

void add_Noise(Mat& src, Mat& dest, bool flag_G = true, bool flag_F = true, double sigma = 20, int num_peppers = 1000) {
	// flag_G=true��������˹����
	// flag_F=true��������������
	dest = Mat::zeros(src.rows, src.cols, src.type()); // ͨ��type()����ͨ����
	RNG rng;
	if (flag_G) {
		rng.fill(dest, RNG::NORMAL, 0, sigma); // ��˹�ֲ�
	}
	dest += src;
	if (flag_F) {
		if (src.channels() == 1) {
			while (num_peppers--) {
				int x = rng.uniform(0, src.rows);
				int y = rng.uniform(0, src.cols);
				if (num_peppers % 2) {
					dest.at<uchar>(x, y) = 0;
				}
				else {
					dest.at<uchar>(x, y) = 255;
				}
			}
		}
		else if (src.channels() == 3) {
			while (num_peppers--) {
				int x = rng.uniform(0, src.rows);
				int y = rng.uniform(0, src.cols);
				if (num_peppers % 2) {
					dest.at<Vec3b>(x, y) = Vec3b(0, 0, 0);
				}
				else {
					dest.at<Vec3b>(x, y) = Vec3b(255, 255, 255);
				}
			}
		}
	}
	return;
}


double get_PSNR(Mat& ori, Mat& noi) {
	double mse, psnr;
	mse = 0;

	for (int i = 0; i < ori.rows; i++) {
		uchar *o = ori.ptr(i);
		uchar *n = noi.ptr(i);
		for (int j = 0; j < ori.cols; j++) {
			mse += (o[j] - n[j]) * (o[j] - n[j]);
		}
	}
	mse /= ori.rows * ori.cols * ori.channels();
	psnr = 10 * log10(255 * 255 / mse);
	return psnr;
}


void NL_means(Mat& src, Mat& dest, int neighborWindowLength, int searchWindowLength, double sigma, double h) {
	// neighborWindowLength should be smaller than searchWindowLength

	if (dest.empty()) {
		//���destΪ�գ�������src��ͬ��С��ͬ�Ϳհ׾������
		dest = Mat::zeros(src.size(), src.type());
	}

	const int half_neighbor_len = neighborWindowLength >> 1;
	const int half_search_len = searchWindowLength >> 1;
	const int boarder_fill = half_neighbor_len + half_search_len;	// ͼƬ��Ե���

	const int neighbor_size = neighborWindowLength * neighborWindowLength;
	const int search_size = searchWindowLength * searchWindowLength;
	const double neighbor_DIV = 1.0 / (double)neighbor_size;	// ���ȷֲ���neighbor��Ȩ��


	vector<double> Gaussian_weight(256 * 256 * src.channels());	// TODO: 256��ʲô��
	double *Gauss_w = &Gaussian_weight[0];
	if (sigma == 0.0) {
		sigma = h;
	}
	double coeff = -1.0 / (double)src.channels() * 1.0 / (h*h);
	int max_id = 0;
	for (int i = 0; i < 512 * 512 * src.channels(); i++) {
		Gauss_w[i] = exp(max(i - 2 * sigma*sigma, 0.0) * coeff);
		if (Gauss_w[i] < 0.001) {
			// ̫С��Ȩ�ؾ����ε�
			max_id = i;
			break;
		}
	}
	for (int i = max_id; i < 256 * 256 * src.channels(); i++) {
		Gauss_w[i] = 0;
	}


	Mat img;
	copyMakeBorder(src, img, boarder_fill, boarder_fill, boarder_fill, boarder_fill, cv::BORDER_REFLECT_101);
	if (src.channels() == 1) {
		size_t neighbor_step = img.step - neighborWindowLength;
		size_t search_step = img.step - searchWindowLength;
		for (int i = 0; i < src.rows; i++) {
			cout << i << '\t';
			uchar *line_ptr = dest.ptr(i);	// ������destͼ��д����ֵ
			for (int j = 0; j < src.cols; j++) {
				// ����src����ʵ���ص㣬ѭ����ȷ��һ�����ص�
				int *similarity = new int[search_size];	//��¼������ÿ��������ƶ�
				double *weight = new double[search_size];	//��¼�������и����Ȩ��
				double weight_sum = 0.0;	// ������ص���������������������Ȩ�غ�

				uchar *P_left_top = img.data + img.step * (half_search_len + i) + (half_search_len + j);	// �ҵ�src[i, j]������P�����Ͻ�
				uchar *Q_left_top = img.data + img.step * i + j;	// �ҵ�src[i, j]���������С����Ͻǵĵ�(�������ĵ�һ�����ص�)��������Q�����Ͻ�
				int searched = 0;	// ��¼�Ѿ��������ĵ���
				for (int m = 0; m < searchWindowLength; m++) {
					for (int n = 0; n < searchWindowLength; n++) {
						// ����src[i, j]��������ѭ����ȷ��һ��������
						int norm = 0;	// �ۼ�����������е��L2 Norm
						uchar *p = P_left_top;	// p�̶�
						uchar *q = Q_left_top + img.step * m + n;	// �ҵ�src[i, j]���������С����ֱ������ĵ㡱������Q�����Ͻ�
						for (int a = 0; a < neighborWindowLength; a++) {
							for (int b = 0; b < neighborWindowLength; b++) {
								// ����src[i, j]������P���ͱ��������������Q��ѭ����ȷ��һ�������
								norm += (*p - *q) * (*p - *q);	// Ϊ���㣬�������������е��Ȩֵ��ͬ�������������Ȩ��
								p++;
								q++;
							}
							p += neighbor_step;
							q += neighbor_step;
						}
						int sim = int (norm * neighbor_DIV); // ���ۼƵ�Normȡƽ��ֵ��Ϊ������������ƶȣ���Ȼ��ԽСԽ����
						similarity[searched++] = sim;
						//get weighted Euclidean distance
						weight_sum += Gauss_w[sim];	// �����ƶ���Ϊ��˹�ֲ���x�ᣬ���ո�˹�ֲ�����Ȩ��
					}
				}

				//���������Ȩ�ع�һ��
				if (weight_sum == 0.0) {
					for (int t = 0; t < search_size; t++) {
						weight[t] = 0;
					}
					weight[(search_size >> 1) + 1] = 1;	// ȡ�����������Լ���ֵ
				}
				else {
					for (int t = 0; t < search_size; t++) {
						weight[t] = Gauss_w[similarity[t]] / weight_sum;
					}
				}
				double ans = 0;
				int added = 0;	// ��ʼ���
				uchar *s = Q_left_top + img.step * half_neighbor_len + half_neighbor_len;	// �ҵ�����������Ͻ�
				for (int a = 0; a < searchWindowLength; a++) {
					for (int b = 0; b < searchWindowLength; b++) {
						ans += (*s) * weight[added++];
						s++;
					}
					s += search_step;
				}

				*line_ptr = saturate_cast<uchar>(ans);
				line_ptr++;
			}
		}
	}
	else if (src.channels() == 3) {

	}
	return;
}



int main() {
	String path = "figs/lena_color.tiff";
	Mat ori = imread(path, 0); // flags=1�����ȡԭͼ��flags=0����ת��Ϊ�Ҷ�ͼ��ȡ
	if (ori.empty()) {
		cout << "Invalid path..." << endl;
		return -1;
	}
	Mat noised_img, output_img;
	double sigma = 20;
	add_Noise(ori, noised_img, true, true, sigma, 1000); // ������˹����,sigma����Ϊ20��������������,���θ�������Ϊ1000

	cout << "raw" << endl;
	cout << "psnr: " << get_PSNR(ori, noised_img) << endl;
	cout << endl;
	imshow("ori", ori);
	imshow("noised_img", noised_img);


	cout << "gaussian filter (7x7) sigma = 5" << endl;
	int64 pre = getTickCount();
	GaussianBlur(noised_img, output_img, Size(7, 7), (5));
	cout << "psnr: " << get_PSNR(ori, output_img) << endl;
	cout << "time: " << 1000 * (getTickCount() - pre) / getTickFrequency() << "ms" << endl;
	cout << endl;
	imshow("gaussian", output_img);


	cout << "median filter (3x3)" << endl;
	pre = getTickCount();
	medianBlur(noised_img, output_img, 3);
	cout << "psnr: " << get_PSNR(ori, output_img) << endl;
	cout << "time: " << 1000 * (getTickCount() - pre) / getTickFrequency() << "ms" << endl;
	cout << endl;
	imshow("median", output_img);


	cout << "bilateral filter(7x7) sigmaColor = 35, sigmaSpace = 5" << endl;
	pre = getTickCount();
	bilateralFilter(noised_img, output_img, 15, 35, 5);
	cout << "psnr: " << get_PSNR(ori, output_img) << endl;
	cout << "time: " << 1000 * (getTickCount() - pre) / getTickFrequency() << "ms" << endl;
	cout << endl;
	imshow("bilateral", output_img);

	cout << "NL means" << endl;
	pre = getTickCount();
	NL_means(noised_img, output_img, 3, 512, sigma, sigma);
	cout << "psnr: " << get_PSNR(ori, output_img) << endl;
	cout << "time: " << 1000 * (getTickCount() - pre) / getTickFrequency() << "ms" << endl;
	cout << endl;
	imshow("nl means", output_img);

	waitKey(0);
	return 0;
}