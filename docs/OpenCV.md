# Question&Tips

读取图像和保存图像的路径如果用相对路径，默认此时是在build文件夹里面 



copyTo()函数有两个，一个在Mat类中，一个在cv命名空间中，在cv命名空间中的函数可以直接使用，而在Mat类中的函数需要首先有一个Mat变量，以`img.copyTo`的形式调用，是不是所有函数都有这样的两种形式（和赫赫讨论，怀疑是版本向下兼容的问题）



# 数字图像

数字图像：是二维图像用有限数字数值像素的表示，由数组或矩阵表示数字图像可以理解为一个二维函数f(x, y)，其中x和y是空间（平面）坐标，而在任意坐标处的幅值f称为图像在该点处的**强度或灰度**

## 图像格式

|        |                                                              |
| ------ | ------------------------------------------------------------ |
| BMP    | Windows系统下的标准位图格式，未经过压缩，一般图像文件会比较大。在在很多软件中被广泛应用 |
| JP(E)G | 也是应用最广泛的图片格式之一，它采用一种特殊的有损压缩算法，达到较大的压缩比（可达到2:1甚至40:1），互联网上最广泛使用的格式 |
| GIF    | 不仅可以是一张静止的图片，也可以是动画，并且支持透明背景图像，适用于多种操作系统，“体型”很小，网上很多小动画都是GIF格式。但是其色域不太广，只支持256种颜色 |
| PNG    | 与JPG格式类似，压缩比高于GIF，支持图像透明，支持Alpha通道调节图像的透明度 |

## 图像尺寸

图像尺寸的长度与宽度是以像素（pixel）为单位

像素是数码影像最基本的单位，每个像素就是一个小点，而不同颜色的点聚集起来就变成照片。
灰度像素点数值范围在0到255之间，0表示黑，255表示白，其它值表示处于黑白之间；
彩色图用红、绿、蓝三通道的二维矩阵来表示每个数值也是在0到255之间，0表示相应的基色，而255则代表相应的基色在该像素中取得最大值

## 分辨率

**每英寸图像内的像素点数**，单位是像素每英寸（PPI）

图像分辨率越高，像素的点密度越高，图像越清晰

## 通道数&位深度

**位深度**

描述图像中每个pixel数值所占的二进制**位数**。位深度越大则图像能表示的颜色数就越多



N通道图：每个像素点都有N个**值**表示，位深度是m，则每个值由m个二进制位表示

8位：单通道图像，也就是灰度图，灰度值范围2\*8=256

24位：三通道3*8=24

32位：三通道加透明度Alpha通道

## 灰度转化

将三通道图（彩色图）转化为单通道图像（灰度图）

## RGB & BGR转化

BGR和RGB的区别在于颜色通道的排列顺序，因为OpenCV使用BGR方式，而大部分图片都是RGB方式，所以需要转化

|      |                                                              |
| ---- | ------------------------------------------------------------ |
| RGB  | 最常见的颜色表示方式，在大多数应用中使用                     |
| BGR  | 在一些特定的应用中使用，例如OpenCV图像处理库中就采用了BGR表示方式 |

## 通道分离/合并

|      |                                                              |
| ---- | ------------------------------------------------------------ |
| 分离 | 将彩色图像，分成B，G，R，3个单通道图像<br />方便我们对BGR三个通道分别进行操作 |
| 合并 | 通道分离为B，G，R后，对单独通道进行修改，最后将修改后的三通道合并为彩色图像 |

## 图像直方图

用以表示数字图像中亮度分布的直方图，标绘了图像中每个亮度值的像素数。

<img src="%E5%9B%BE%E7%89%87/image-20230907142412706.png" alt="image-20230907142412706" style="zoom:67%;" />

横轴是灰度值（0~255），纵轴是像素数量

图像直方图的意义：

直方图是图像中像素强度分布的图形化表达

它统计了每一个灰度值所具有的像素个数

CV领域常借助图像直方图来实现图像的二值化

## 颜色空间

是一种用于描述和表示颜色的数学模型，说白了就是一个描述颜色的方式，因为我们说这个颜色是红色，有多红，是斩男红还是



# Mat对象

<img src="%E5%9B%BE%E7%89%87/image-20230907195853977.png" alt="image-20230907195853977" style="zoom:67%;" />

注：关于矩阵头中的数据类型

Mat就是一个数据类型，这里的数据类型是指Mat这个矩阵里到底是什么数据类型，是opencv规定的数据类型，包括CV_32F，CV_8U，不再使用int，double，避免不同系统int的位数不同

## 创建

利用矩阵**宽高**和**数据类型**创建Mat类

```c++
Mat(int rows, int cols, int type)

//例:
Mat a(3, 3, CV_8UC1);
```

利用矩阵**Size()结构**和**数据类型**创建Mat类

```
Mat(Size size, int type)
size: Size(cols, rows)
例:
Mat a(Size(4, 4), CV_8UC1)
```

利用已有Mat类创建新的Mat类（抠图）

![image-20230801180752581](%E5%9B%BE%E7%89%87/image-20230801180752581.png)

`Range(2, 4)`实际是左闭右开区间，[2, 4)取到2和3



## 赋值

创建时赋值

```
Mat(int rows, int cols, int type, const Scalar& s)
s: 给矩阵中每个像素赋值的参数变量，例Scalae(0,0,255)
```

类方法（函数）赋值

|       |                 |
| ----- | --------------- |
| eye   | 单位矩阵        |
| diag  | 角矩阵          |
| ones  | 元素全为1的矩阵 |
| zeros | 元素全为0的矩阵 |

```c++
Mat::类方法(rows, cols, type);
或 Mat::类方法(Size(cols, rows), type);

//例:
m2 = Mat::ones(3, 3, CV_8UC3);
```

枚举法赋值

```
Mat a = (Mat_<int>(3,3)<<1,2,3,4,5,6,7,8,9);
Mat b = (Mat_<double>(2,3)<<1.0,2.1,3.2,4.0,5.1,6.2)
```



```c++
Mat m1, m2;
m1 = image.clone();//深拷贝
image.copyto(m2);//浅拷贝
m3 = image//浅拷贝
```

## Mat类数据读取

![image-20230803233853933](%E5%9B%BE%E7%89%87/image-20230803233853933.png)

常用属性

![image-20230803233957997](%E5%9B%BE%E7%89%87/image-20230803233957997.png)

Mat元素的读取

![image-20230804092900722](%E5%9B%BE%E7%89%87/image-20230804092900722.png)

## 运算



![image-20230804094550147](%E5%9B%BE%E7%89%87/image-20230804094550147.png)

# ==读取、显示、保存、深拷贝、像素值==

### 读取imread()

image read

`imread(const string& path, int flag)`

flag：确定读取图像形式（IMREAD_COLOR，IMREAD_GRAYSCALE）

其中path路径中不要出现中文，否则会报错

### 显示imshow()

`imshow(const string& winname, InputArray mat)`

winname：就是创建的窗口的名字

mat：要显示的图像矩阵



### namedWindow()

>为了避免程序不能自动释放窗口内存，可以不使用namedWindow()
>
>直接就用imshow()，否则需要用distory函数去主动释放

`namedWindow(const string& winName, int flag = WINDOW_AUTOSIZE)`

winname：窗口名称，用作窗口的标识符

flag：窗口属性设置标志

| flag            |                |
| --------------- | -------------- |
| WINDOW_AUTOSIZE | 默认尺寸       |
| WINDOW_NORMAL   | 可任意调整尺寸 |

```
    Mat img = imread("../binary.jpg");
    namedWindow("binary_1", WINDOW_FREERATIO);
    imshow("binary_1", binary_1);
    waitKey(0);
    destroyAllWindows();
```

### waitKey()

waitKey(delay)在一个给定的时间内(单位ms)等待用户按键触发

有按键按下，返回按键的ASCII值。无按键按下，返回-1。

waitKey(100),表示程序每100ms检测一次按键，检测到返回按键值，检测不到返回-1；

### 保存imwrite()

读取的图像文件格式和保存下来的格式可以不一样

![image-20230804100004176](%E5%9B%BE%E7%89%87/image-20230804100004176.png)

params是对图片的压缩形式

```c++
imwrite("new_pic", img);
```

### copyTo()

```
img.copyTo(src);
```



## 操作像素点

### ptr\<uchar>

`ptr<uchar>()`是一个成员函数，属于OpenCV库的`cv::Mat`类。该函数返回一个指向图像某一行首元素的指针，可以方便地访问和修改图像中的每个像素。

```
uchar* row_ptr = img.ptr<uchar>(i); //获取第i行的首地址
uchar pixel_value = row_ptr[j]; // 获取第j列像素的值
```



### at()

at()函数是对像素点进行操作

返回值为`<>`中的类型

Vec3b ---> uchar类型，长度为3的vector数组 

```
//方式3：at()函数操作（逐像素访问）
void At()
{
	//彩色图
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)				//逐像素
		{
			src.at<Vec3b>(i, j)[0] = 255 - (int)src.at<Vec3b>(i, j)[0];		//逐通道
			src.at<Vec3b>(i, j)[1] = 255 - (int)src.at<Vec3b>(i, j)[1];		//逐通道
			src.at<Vec3b>(i, j)[2] = 255 - (int)src.at<Vec3b>(i, j)[2];		//逐通道
		}
	}
 
	//灰度图
	for (int i = 0; i < src_gray.rows; i++)
	{
		for (int j = 0; j < src_gray.cols; j++)
		{
			src_gray.at<uchar>(i, j) = 255 - (int)src_gray.at<uchar>(i, j);
		}
	}
}
```



一、单通道图像
对于单通道图像"picture"，picture.at(i,j)就表示在第i行第j列的像素值。
即读取了位于(i,j)的像素值

二、多通道图像
对于多通道图像如RGB图像"picture"，可以用picture.at(i,j)[c]来表示某个通道中在(i,j)位置的像素值。

1）上面的double、Vec3b表示图像元素的类型（ Vec3b 表示图像的像素类型，它是一个长度为 3 的 uchar（无符号字符）数组，用于表示三通道的彩色图像像素点）

2）(i,j)当然就是指像素点的位置，表示第i行第j列。

3）[c]表示的是通道，对于RGB图像而言，c取0就是B分量；c取1就是G分量；c取2就是R分量（要注意在OpenCV中是按BGR的顺序表示的）。

```
cv::Vec3b pixelColor = image.at<cv::Vec3b>(100, 200);
	    // 访问 B、G、R 三个通道的值
	    uchar blue = pixelColor[0]; // 蓝色通道
	    uchar green = pixelColor[1]; // 绿色通道
	    uchar red = pixelColor[2]; // 红色通道
```



# 视频加载&摄像头调用

VideoCapture()

![image-20230804234110062](%E5%9B%BE%E7%89%87/image-20230804234110062.png)

```c++
VideoCapture video;
video.open(path);
video.isOpened();
```

利用get()函数获得视频的属性

![image-20230804234523954](%E5%9B%BE%E7%89%87/image-20230804234523954.png)

```
VideoCapture video;
video.get(CAP_PROP_...);
```

## 显示

将视频帧读取到Mat矩阵中，有两种方式：一种是read()操作；另一种是 “>>”操作

```
Mat frame;
cap.read(frame);
cap >> frame;
```



## 保存

VideoWriter()

![image-20230804235308102](%E5%9B%BE%E7%89%87/image-20230804235308102.png)

# ==颜色空间变换==

## RGB

常见的图像数据类型

8U：0~255

32F：0~1

64F：0~1

对于后两种

大于1-->白色

小于0-->黑色

[0, 1]映射到[0, 255]

那么，为什么要将[0, 255]缩为[0, 1]呢，因为图像有乘除运算，会产生小数，如果舍入成整数，又会导致误差太大

## 数据类型变换

![image-20230907155752407](%E5%9B%BE%E7%89%87/image-20230907155752407.png)

`convertTo(OutputArray m, int rtype, double alpha, double beta)`

m：输出图像

rtype：目标深度

alpha：缩放系数

beta：平移系数

公式：$\alpha*I(x,y)+\beta$

```c++
//使用方法CV_8C1-->CV_32FC1
Mat old_img;//原图像
Mat new_img;
a.convertTo(new_img, CV_32F, 1.0/255, 0);
```



关于CV_32F和CV_32FC1的区别

一句话，应指定通道数时，用`CV_32FCx`. 如果只需要深度，用`CV_32F`

[C++ OpenCV 类型 CV_32F 和 CV_32FC1 的区别-IGI (igiftidea.com)](https://www.igiftidea.com/article/10979087775.html)

## 灰度图--彩色图变换cvtColor()

![image-20230907163358814](%E5%9B%BE%E7%89%87/image-20230907163358814.png)

```c++
Mat img;
Mat HSV;
cvtColor(img, HSV, COLOR_BGR2HSV);
```

```
Mat img = imread("../binary.jpg");
Mat grey;
cvtColor(img, grey, COLOR_BGR2GRAY);
```

## 通道分离split()

```
//vector & 数组 形式
split(InputArray m, OutputArrayOfArrays mv);
```

m：待分离的多通道图像

mv：分离后的单通道图像，为向量vector形式

```
cv::Mat bgr[3];
cv::split(src, bgr);
```

## 通道合并merge()

```
//vector形式
merge(InputArrayOfArrays mv, OutputArray dst)
//数组形式
merge(InputArrayOfArrays mv, int num, OutputArray dst)
```

mv：需要合并的图像向量vector，其中每个图像必须拥有相同的尺寸和数据类型

dst：合并后输出的图像，通道数等于所有输出图像的通道数总和

num：数组的大小，就是通道数

```
    Mat img = imread("../001.jpg", IMREAD_COLOR);
    imshow("原图", img);
    waitKey(0);

    Mat arr[3];
    split(img, arr);
    Mat img_1 = arr[0];
    Mat img_2 = arr[1];
    Mat img_3 = arr[2];
	
	Mat new_split_img;
    merge(arr, 3, new_split_img);
    imshow("new_split_img", new_split_img);
    waitKey(0);
    
    Mat zero = Mat::zeros(img.rows, img.cols, CV_8UC1);
    vector<Mat> vec;
    vec.push_back(img_1);
    vec.push_back(zero);
    vec.push_back(zero);
    Mat look;
    merge(vec, look);

    imshow("split arr[3]", look);
    waitKey(0);
```



## 图像二值化threshold()

![image-20230914095832804](%E5%9B%BE%E7%89%87/image-20230914095832804.png)

maxval根据type来决定是否使用，如果 不使用，可以填写任意值

```c++
    Mat img = imread("../binary.jpg");
    Mat grey;
    cvtColor(img, grey, COLOR_BGR2GRAY);
    Mat binary_1;
    threshold(grey, binary_1, 125, 255, THRESH_BINARY);
    namedWindow("binary_1", WINDOW_FREERATIO);
    imshow("binary_1", binary_1);
    waitKey(0);

    Mat binary_2;
    threshold(grey, binary_2, 125, 255, THRESH_BINARY_INV);
    namedWindow("binary_2", WINDOW_FREERATIO);
    imshow("binary_2", binary_2);
    waitKey(0);
    destroyAllWindows();
```

类似于滤波的感觉，图中的蓝色线就是阈值

![image-20230914095625800](%E5%9B%BE%E7%89%87/image-20230914095625800.png)



# ==尺寸变换==

## 缩放resize()

![image-20230915104848032](%E5%9B%BE%E7%89%87/image-20230915104848032.png)

dsize参数和fx,fy参数二选一即可，如果矛盾以dsize为准

| interpolation(插值方法)     |
| --------------------------- |
| INTER_AREA (区域插值)       |
| INTER_CUBIC (三次样条插值)  |
| INTER_LINEAR (线性插值)     |
| INTER_NEAREST (最近邻插值） |

[OpenCV图像缩放resize各种插值方式的比较_opencv resize_AI吃大瓜的博客-CSDN博客](https://blog.csdn.net/guyuealian/article/details/85097633/?ops_request_misc=&request_id=&biz_id=102&utm_term=INTER_AREA&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-85097633.142^v94^insert_down28v1&spm=1018.2226.3001.4187)

- 速度比较：INTER_NEAREST（最近邻插值)>INTER_LINEAR(线性插值)>INTER_CUBIC(三次样条插值)>INTER_AREA  (区域插值)
- 对图像进行缩小时，为了避免出现波纹现象，推荐采用INTER_AREA 区域插值方法。
- OpenCV推荐：如果要缩小图像，通常推荐使用INTER_AREA插值效果最好，而要放大图像，通常使用INTER_CUBIC(速度较慢，但效果最好)，或者使用INTER_LINEAR(速度较快，效果还可以)。至于最近邻插值INTER_NEAREST，一般不推荐使用

```c++
	Mat grey = imread("../binary.jpg", IMREAD_GRAYSCALE);
    Mat dst, smal_1, smal_2, smal_3, smal_4;
    resize(grey, dst, Size(30, 30), 0, 0, INTER_AREA);//先缩小
    resize(dst, smal_1, Size(50, 50), 0, 0, INTER_LINEAR);//用三种不同的方式放大
    resize(dst, smal_2, Size(50, 50), 0, 0, INTER_CUBIC);
    resize(dst, smal_3, Size(50, 50), 0, 0, INTER_NEAREST);

    namedWindow("smal_1", WINDOW_FREERATIO);
    imshow("smal_1", smal_1);
    namedWindow("smal_2", WINDOW_FREERATIO);
    imshow("smal_2", smal_2);
    namedWindow("smal_3", WINDOW_FREERATIO);
    imshow("smal_3", smal_3);
    waitKey(0);
    destroyAllWindows();
```

## ROI区域裁剪Rect()

```
Rect::Rect(
   int x,      // 矩形左上角的 x 坐标
   int y,      // 矩形左上角的 y 坐标
   int width,  // 矩形的宽度
   int height  // 矩形的高度
) 
```







## 翻转flip()

![image-20230915105334498](%E5%9B%BE%E7%89%87/image-20230915105334498.png)

```c++
	Mat fl_x, fl_y, fl_xy;
    flip(grey, fl_x, 0);
    flip(grey, fl_y, 1);
    flip(grey, fl_xy, -1);
    imshow("fl_x", fl_x);
    imshow("fl_y", fl_y);
    imshow("fl_xy", fl_xy);
    waitKey(0);
```

## 拼接h/vconcat()

![image-20230915105424180](%E5%9B%BE%E7%89%87/image-20230915105424180.png)

```
	Mat grey = imread("../001.jpg", IMREAD_GRAYSCALE);
    
    Mat fl_x, fl_y, fl_xy;
    flip(grey, fl_x, 0);
    flip(grey, fl_y, 1);
    flip(grey, fl_xy, -1);
    
    Mat img_1, img_2, img;
    hconcat(grey, fl_y, img_1);
    hconcat(fl_x, fl_xy, img_2);
    vconcat(img_1, img_2, img);
    
    namedWindow("img", WINDOW_FREERATIO);
    imshow("img", img);
    waitKey(0);
    destroyAllWindows();
```

# 仿射变换

仿射变换是由：平移、缩放、旋转翻转和错切组合得到，也称为三点变换

<img src="%E5%9B%BE%E7%89%87/image-20230917164937779.png" alt="image-20230917164937779" style="zoom: 67%;" />

## 仿射变换函数

![image-20230917165041531](%E5%9B%BE%E7%89%87/image-20230917165041531.png)

边界填充方法最常用的就是默认值，因为我们不关心填充的数据，而且黑色不影响其他信息的表达

## 计算旋转矩阵

没有单独的旋转函数，需要先得到旋转的仿射矩阵，再调用wrapAffine()函数

![image-20230917170408828](%E5%9B%BE%E7%89%87/image-20230917170408828.png)

## 计算仿射矩阵

![image-20230917170714003](%E5%9B%BE%E7%89%87/image-20230917170714003.png)

可以说getRotationMatrix2D()是一种计算仿射矩阵的特殊方式，而getAffineTransform()是通过三点坐标对应来计算，更加一般化



```c++
    Mat img = imread("../001.jpg");
    if(img.empty()){
        cout << "error" << endl;
    }
	Mat rotation_matrix;
    Point2f center(img.cols / 2.0, img.cols / 2.0);
    rotation_matrix = getRotationMatrix2D(center, 45, 1);
    Mat rotation_img;
    warpAffine(img, rotation_img, rotation_matrix, Size(img.cols, img.rows));
    imshow("rotation_img", rotation_img);
    waitKey(0);

    Mat Aiffine_matrix;
    Point2f src[3];
    src[0] = Point2f(0, 0);
    src[1] = Point2f(0, (float)(img.cols-1));
    src[2] = Point2f((float)img.rows-1, (float)(img.cols-1));

    Point2f dst[3];
    dst[0] = Point2f((float)img.rows * 0.11, (float)img.cols * 0.20);
    dst[1] = Point2f((float)img.rows * 0.35, (float)img.cols * 0.68);
    dst[2] = Point2f((float)img.rows * 0.77, (float)img.cols * 0.90);
    Aiffine_matrix = getAffineTransform(src, dst);
    Mat Aiffine_img;
    warpAffine(img, Aiffine_img, Aiffine_matrix, Size(img.cols, img.rows));
    imshow("Aiffine_img", Aiffine_img);
    waitKey(0);

```

# 透视变换

即透视投影

<img src="%E5%9B%BE%E7%89%87/image-20230917203213568.png" alt="image-20230917203213568" style="zoom:50%;" />

## 计算透视变换矩阵

通过原图像和变换后图像的4个对应点，可以求得变换矩阵

![image-20230917203046278](%E5%9B%BE%E7%89%87/image-20230917203046278.png)

![image-20230917203312807](%E5%9B%BE%E7%89%87/image-20230917203312807.png)

## 透视变换函数

![image-20230917203457686](%E5%9B%BE%E7%89%87/image-20230917203457686.png)



```
	Mat img = imread("../book.jpg");
    if(img.empty()){
        cout << "error" << endl;
    }
    imshow("img", img);
    waitKey(0);
    Mat per_matrix;
    Point2f src[4];
    src[0] = Point2f((float)img.cols * 0.25, 0.0);//左上角
    src[1] = Point2f(0.0, (float)img.rows);//左下角
    src[2] = Point2f((float)img.cols * 0.75, 0.0);//右上角
    src[3] = Point2f((float)img.cols, (float)img.rows);//右下角

    Point2f dst[4];
    dst[0] = Point2f(0.0, 0.0);
    dst[1] = Point2f(0.0, (float)img.rows);
    dst[2] = Point2f((float)img.cols, 0.0);
    dst[3] = Point2f((float)img.cols, (float)img.rows);

    per_matrix = getPerspectiveTransform(src, dst);
    Mat per_img;
    warpPerspective(img, per_img, per_matrix, Size(img.cols, img.rows));
    namedWindow("per_img", WINDOW_FREERATIO);
    imshow("per_img", per_img);
    waitKey(0);
```

# 绘制图形

## 直线

![image-20230918091315516](%E5%9B%BE%E7%89%87/image-20230918091315516.png)

thickness就是直线的宽度

## 圆形

![image-20230918093046933](%E5%9B%BE%E7%89%87/image-20230918093046933.png)

涉及到封闭图形，如果thickness = -1，就会将其填充

## 矩形

![image-20230918094918096](%E5%9B%BE%E7%89%87/image-20230918094918096.png)

## 文字

![image-20230918095203313](%E5%9B%BE%E7%89%87/image-20230918095203313.png)





# 图像金字塔

![image-20230920205559183](%E5%9B%BE%E7%89%87/image-20230920205559183.png)

如果我们通过下采样生成一个金字塔，最简单的做法就是：不断地删除图像的偶数行和偶数列，重复这个过程，就得到一个金字塔。
如果我们通过上采样生成一个金字塔，最简单的就是：在每列像素点的右边插入值为0的列，在每行像素点下面插入值为0的行，不断重复，就生成一个金字塔了。但是这些0值像素点毫无意义，我们就需要对0值像素点进行赋值。而赋值就是插值处理。插值处理也有很多方法，比如用区域**均值补充**，那生成的就是**平均金字塔**，如果用**高斯核**填充就是**高斯金字塔**。

## 高斯金字塔

下采样函数

![image-20230920164035666](%E5%9B%BE%E7%89%87/image-20230920164035666.png)

## 拉普拉斯金字塔

拉普拉斯金字塔是在高斯金字塔的基础上生成的。

第n层拉普拉斯图像实际上是第n层高斯图像与第n+1层高斯图像经上采样后的差值

为啥要发明拉普拉斯金字塔？

因为高斯核填充会导致纹理信息被忽略，而这些丢失的信息就是拉普拉斯金字塔 。所以拉普拉斯金字塔的作用就在于能够恢复图像的细节，就是我们从高层的尺寸小的特征图中提取特征后，我们还能通过拉普拉斯金字塔数据找回高层像素点对应的底层清晰度更高的图像，就是返回来找到更多图像的细节。

`Li = Gi - PyrUp( PyrDown(Gi) )`

![image-20230920164428687](%E5%9B%BE%E7%89%87/image-20230920164428687.png)



# 卷积

通过应用不同的卷积核，可以对图像进行平滑和模糊处理。平滑和模糊可以消除图像中的噪声和细节，使图像变得更加柔和和模糊，从而提高图像的质量和观感。

边缘检测：通过应用特定的卷积核，可以检测图像中的边缘。边缘是图像中亮度变化较大的区域，边缘检测可以帮助我们分析图像的结构和形状，从而实现目标检测、图像分割等应用。



![image-20230922235843961](%E5%9B%BE%E7%89%87/image-20230922235843961.png)

一般来说，卷积核中所有元素之和应该等于 1 ，因为这样经过卷积运算之后，图像的亮度保持不变

可以看到卷积核的尺寸越大，图像就越模糊

高斯内核，距离中心越近，权值越大。高斯模糊在平滑物体表面的同时，能够更好的保持图像的边缘和轮廓。（但是，这是为什么呢？？？）

[卷积神经网络 —— 图像卷积_xuechanba的博客-CSDN博客](https://blog.csdn.net/xuechanba/article/details/125080692?ops_request_misc=%7B%22request%5Fid%22%3A%22169539765416800184186855%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=169539765416800184186855&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-125080692-null-null.142^v94^insert_down28v1&utm_term=图像卷积&spm=1018.2226.3001.4187)





# 噪声

图像在获取或者传输过程中会受到随机信号的干扰产生噪声。

椒盐噪声：又被称作脉冲噪声，它会随机改变图像中的像素值，是由相机成像、图像传输、解码处理等过程产生的**黑白**相间的亮暗点噪声，就像胡椒面和盐粒

高斯噪声：高斯噪声是指噪声分布的概率密度函数服从高斯分布（正态分布）的一类噪声







# 滤波

## 均值滤波





# ==形态学处理==

> 主要是对二值化后的图像处理
>
> 膨胀腐蚀操作的本质是对图像进行卷积操作



生成结构元素的函数

```
cv::getStructuringElement 函数生成，参数如下
    cv::Mat cv::getStructuringElement(
    int         shape,      // 核的形状 (具体类型可以参考网络资料，或者利用 vscode 的代码提示)
    Size        ksize,      // 核的大小
    Point       anchor = Point(-1,-1) // 锚点
)
```

![image-20231122195944640](%E5%9B%BE%E7%89%87/image-20231122195944640.png)

## 腐蚀erode()

什么是腐蚀？腐蚀就是将图像中的白色部分缩小，黑色部分扩大，这样就可以去除小的白色区域

```
void cv::erode(
    InputArray  src,        // 输入的二值图像
    OutputArray dst,        // 输出的二值图像
    InputArray  kernel,     // 腐蚀的核
    Point       anchor = Point(-1,-1), // 锚点,中心点在结构元素中的位置，默认参数为结构元素的几何中心点
    int         iterations = 1,        // 腐蚀的次数
    int         borderType = BORDER_CONSTANT, // 边界类型
    const Scalar& borderValue = morphologyDefaultBorderValue() // 边界值
    )
```



## 膨胀dilate()

什么是膨胀？膨胀就是将图像中的白色部分扩大，黑色部分缩小，这样就可以去除小的黑色区域

```
void cv::dilate(
    InputArray  src,        // 输入的二值图像
    OutputArray dst,        // 输出的二值图像
    InputArray  kernel,     // 膨胀的核
    Point       anchor = Point(-1,-1), // 锚点
    int         iterations = 1,        // 迭代次数
    int         borderType = BORDER_CONSTANT, // 边界类型
    const Scalar& borderValue = morphologyDefaultBorderValue() // 边界值
    )
```

# ==轮廓==

## 边缘检测原理

![image-20231127194903953](%E5%9B%BE%E7%89%87/image-20231127194903953.png)

计算梯度有不同的算子



## 轮廓层次结构

<img src="%E5%9B%BE%E7%89%87/image-20231127202232801.png" alt="image-20231127202232801" style="zoom:80%;" />



## 轮廓检测findContours()

```
void cv::findContours(
    InputOutputArray    image,          // 输入的二值图像
    OutputArrayOfArrays contours,       // 输出的轮廓存放的像素坐标
    OutputArray         hierarchy,      // 输出的轮廓的层次结构
    int                 mode,           // 轮廓的检索模式 (具体类型可以参考网络资料，或者利用 vscode 的代码提示)
    int                 method,         // 轮廓的近似方法 (具体类型可以参考网络资料，或者利用 vscode 的代码提示)
    Point               offset = Point() // 偏移量，主要用于得到 ROI区域找到的轮廓 在整个图像中的位置
)
```

[学习笔记：C++环境下OpenCV的findContours函数的参数详解及优化_c++ opencv findcontours-CSDN博客](https://blog.csdn.net/KarvinDish/article/details/125425123?ops_request_misc=%7B%22request%5Fid%22%3A%22170115296316800188535644%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170115296316800188535644&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-125425123-null-null.142^v96^pc_search_result_base7&utm_term=findcontours c%2B%2B&spm=1018.2226.3001.4187)

## 轮廓绘制drawContours()

```
void cv::drawContours(
    InputOutputArray    image,          // 输入的图像
    InputArrayOfArrays  contours,       // 输入的轮廓
    int                 contourIdx,     // 要绘制的轮廓的索引,如果是负数，绘制所有的轮廓
    const Scalar&       color,          // 轮廓的颜色
    int                 thickness = 1,  // 轮廓的粗细
    int                 lineType = LINE_8, // 轮廓的线型
    InputArray          hierarchy = noArray(), // 轮廓的层次结构
    int                 maxLevel = INT_MAX, // 轮廓的最大层次
    Point               offset = Point() // 偏移
)
```



# 流程

```
	vector<vector<cv::Point>> contours; // 定义一个二维数组，用于存放轮廓
    vector<cv::Vec4i> hierarchy; // 定义一个一维数组，用于存放轮廓的层次结构
    
    cv::threshold(img, binary, 140, 255, cv::THRESH_BINARY); // 二值化
    cv::findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0)); // 查找轮廓
    cv::drawContours(drawContours, contours, -1, cv::Scalar(0, 255, 255), 3); // 绘制轮廓
```



## 轮廓面积contourArea()

```
double cv::contourArea(
        InputArray contour, // 输入的轮廓
        bool       oriented = false // 是否为有向面积
)
```

## 外接矩形&绘制

```c++
//最大外接矩形
Rect cv::boundingRect(
        InputArray points // 输入的轮廓: 灰度图像、二值化图像、轮廓坐标o
)

void cv::rectangle(
         InputOutputArray img, // 输入的图像
         Rect              rec, // 矩形
         const Scalar&     color, // 矩形的颜色
         int               thickness = 1, // 矩形的粗细
         int               lineType = LINE_8, // 矩形的线型
         int               shift = 0 // 矩形的偏移
)

for (int i = 0; i < contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        cv::rectangle(demo, rect, cv::Scalar(0, 255, 255), 2); // 绘制轮廓外接矩形

}    
    
//最小外接矩形
RotatedRect cv::minAreaRect(
        InputArray points   
)

void cv::line(
        InputOutputArray img, // 输入的图像
        Point             pt1, // 线段的起点
        Point             pt2, // 线段的终点
        const Scalar&     color, // 线段的颜色
        int               thickness = 1, // 线段的粗细
        int               lineType = LINE_8, // 线段的线型
        int               shift = 0 // 线段的偏移
)

for (int i = 0; i < contours.size(); i++) {
        cv::RotatedRect rrect = cv::minAreaRect(contours[i]); // 获取轮廓的最小外接矩形
        cv::Point2f pts[4]; // 定义一个数组，用于存放旋转矩形的4个顶点
        rrect.points(pts); // 获取旋转矩形的4个顶点
        for(int j = 0; j < 4; j++){
            cv::line(demo, pts[j], pts[(j + 1) % 4], cv::Scalar(0, 255, 255), 2); // 绘制轮廓最小外接矩形(这里是绘制4条线，即旋转矩形的4条边),如果
        }
        cv::putText(demo, "minAreaRect", cv::Point(rrect.center.x, rrect.center.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    }
```





