#include <iostream>
#include <string>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/matrix.h>
#include <dlib/geometry/vector.h>
#include <dlib/dnn.h>
#include <dlib/pixel.h>

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>


using namespace dlib;
using namespace std;

//typedef matrix<double,0,1> cv;


// this code is copyed from dlib python interface

class face_recognition_model_v1
{

public:

    face_recognition_model_v1(const std::string& model_filename)
    {
        deserialize(model_filename) >> net;
    }

    matrix<double,0,1> compute_face_descriptor (
            matrix<rgb_pixel> img,
            const full_object_detection& face,
            const int num_jitters
    )
    {
        std::vector<full_object_detection> faces(1, face);
        return compute_face_descriptors(img, faces, num_jitters)[0];
    }

    std::vector<matrix<double,0,1>> compute_face_descriptors (
            matrix<rgb_pixel> img,
    const std::vector<full_object_detection>& faces,
    const int num_jitters
    )
    {

        for (auto& f : faces)
        {
            if (f.num_parts() != 68 && f.num_parts() != 5)
                throw dlib::error("The full_object_detection must use the iBUG 300W 68 point face landmark style or dlib's 5 point style.");
        }


        std::vector<chip_details> dets;
        for (auto& f : faces)
            dets.push_back(get_face_chip_details(f, 150, 0.25));
        dlib::array<matrix<rgb_pixel>> face_chips;
        extract_image_chips(img, dets, face_chips);

        std::vector<matrix<double,0,1>> face_descriptors;
        face_descriptors.reserve(face_chips.size());

        if (num_jitters <= 1)
        {
            // extract descriptors and convert from float vectors to double vectors
            for (auto& d : net(face_chips,16))
                face_descriptors.push_back(matrix_cast<double>(d));
        }
        else
        {
            for (auto& fimg : face_chips)
                face_descriptors.push_back(matrix_cast<double>(mean(mat(net(jitter_image(fimg,num_jitters),16)))));
        }

        return face_descriptors;
    }

private:

    dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img,
    const int num_jitters
    )
    {
        std::vector<matrix<rgb_pixel>> crops;
        for (int i = 0; i < num_jitters; ++i)
            crops.push_back(dlib::jitter_image(img,rnd));
        return crops;
    }


    template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

    template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

    template <int N, template <typename> class BN, int stride, typename SUBNET>
    using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

    template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
    template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

    template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
    template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
    template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
    template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
    template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

    using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                                                 alevel0<
                                                         alevel1<
                                                                 alevel2<
                                                                         alevel3<
                                                                                 alevel4<
                                                                                         max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                                                                                         input_rgb_image_sized<150>
                                                                         >>>>>>>>>>>>;
    anet_type net;
};

//draw landmarks over face
void drawFaceLandmarks(cv::Mat &image, full_object_detection faceLandmark){

    //Loop over all face landmarks
    for(int i=0; i< faceLandmark.num_parts(); i++){
        int x = faceLandmark.part(i).x();
        int y = faceLandmark.part(i).y();
        string text = to_string(i+1);

        //Draw a small circle at face landmark over the image using opencv
	cv::circle(image, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA );

        //Draw text at face landmark to show index of current face landmark over the image using opencv
	cv::putText(image, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 0), 1);
    }
}

// Write landmarks to a file
void writeFaceLandmarkstoFile(string facelandmarkFileName, full_object_detection faceLandmark){
    //Open file
    std::ofstream outputFile;
    outputFile.open(facelandmarkFileName);

    //Loop over all face landmarks
    for(int i=0; i<faceLandmark.num_parts(); i++){
        outputFile<<faceLandmark.part(i).x()<<" "<<faceLandmark.part(i).y()<<endl;
    }

    //close file
    outputFile.close();
}

// the main code of c++ compute_face_descriptor
int main(int argc, char ** argv) {
    
    // Adding Visualisation Windows
    image_window win, win_faces;

    // declaring global variables	
    int eye_centre_x_left, eye_centre_y_left,eye_centre_x_right, eye_centre_y_right;
	
    // test for the same image, with only one face
    std::string img_path = "/home/varunsakunia/dlib_test/id011.jpg";

    // Read image using opencv
    cv::Mat Image = cv::imread(img_path);
    cv::Mat inputImage = Image.clone();

    //Check if the mentioned image exits
    if(Image.empty()){
	    std::cout<<"image doesn't exist"<< std::endl;
        return -1;
    }

    // face detector
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    std::string sp_path = "/home/varunsakunia/dlib_test/shape_predictor_5_face_landmarks.dat";
    dlib::shape_predictor sp;
    dlib::deserialize(sp_path) >> sp;

    std::string face_rec_path = "/home/varunsakunia/dlib_test/dlib_face_recognition_resnet_model_v1.dat";
    face_recognition_model_v1 face_encoder = face_recognition_model_v1(face_rec_path);

    // Now we will go ask the shape_predictor to tell us the pose of
    // each face we detected.
    std::vector<dlib::full_object_detection> shapes;

    // converting image to dlib format and reading the image
    dlib::matrix<dlib::rgb_pixel> img;
    dlib::load_image(img, img_path);

    std::vector<dlib::rectangle> dets = detector(img, 1);
    std::cout << "Number of faces detected: " << dets.size() << std::endl;
    //  Number of faces detected: 1

    // Face detector
    //dlib::full_object_detection shape = sp(img, dets[0]);  // only one face
    
    // Now we will go ask the shape_predictor to tell us the pose of each face we detected.
    for (unsigned long j = 0; j < dets.size(); ++j)
            {
		dlib::full_object_detection shape = sp(img, dets[j]);
		std::cout << "Number of parts detected by dlib: "<< shape.num_parts() << std::endl;
     		
		// Finding the Eye-centres
		eye_centre_x_left = ((shape.part(2) + shape.part(3))/2).x();
                eye_centre_y_left = ((shape.part(2) + shape.part(3))/2).y();
                eye_centre_x_right = ((shape.part(0) + shape.part(1))/2).x();
                eye_centre_y_right = ((shape.part(0) + shape.part(1))/2).y();
		std::cout << "Eye Centres (L-R): " << (shape.part(2) + shape.part(3))/2 <<" "<< (shape.part(0) + shape.part(1))/2 << std::endl;
		
		// compute face vector
	    	std::cout << "Face Vector: " << trans(face_encoder.compute_face_descriptor(img, shape, 1)) << std::endl;
		
        	//Draw landmarks on image
		drawFaceLandmarks(inputImage, shape);

        	//Write face Landmarks to a file on disk to analyze
        	string landmarksFilename = "face_landmarks_new_" + to_string(j+1) + ".txt";
        	writeFaceLandmarkstoFile(landmarksFilename, shape);

                // You get the idea, you can get all the face part locations if
                // you want them.  Here we just store them in shapes so we can
                // put them on the screen.
                shapes.push_back(shape);
            }

     // Now let's view our face poses on the screen.
     win.clear_overlay();
     win.set_image(img);
     win.add_overlay(render_face_detections(shapes));

    // We can also extract copies of each face that are cropped, rotated upright, and scaled to a standard size as shown here:
    dlib::array<array2d<rgb_pixel>> face_chips;
    extract_image_chips(img, get_face_chip_details(shapes), face_chips);
    win_faces.set_image(tile_images(face_chips));

    //saving face chip
    int image_id = 2;
    for (int idx = 0; idx < face_chips.size(); idx++){
                std::string fname = "detected_facechip_" + std::to_string(image_id) + ".jpg";
                dlib::save_jpeg(face_chips[idx], fname);
                image_id++;
            }

    //Save image with face landmarks drawn to disk to analyze using opencv
    std:: string imageFileName = "image_face_landmark.jpg";
    cv::imwrite(imageFileName,inputImage);

    //####################(  Draw Eye Centres  )########################
    cv::Point centerCircle_left(eye_centre_x_left, eye_centre_y_left);
    int radiusCircle = 3;
    cv::Scalar colorCircle(0,0,255);
    int thicknessCircle = 2;
    cv::circle(inputImage, centerCircle_left, radiusCircle, colorCircle, thicknessCircle);
    cv::Point centerCircle_right(eye_centre_x_right, eye_centre_y_right);
    cv::circle(inputImage, centerCircle_right, radiusCircle, colorCircle, thicknessCircle);

    //Create windows to show images
    cv::namedWindow("Input image", cv::WINDOW_NORMAL);
    cv::namedWindow("Output image", cv::WINDOW_NORMAL);

    //Display both images with face landmarks and without face landmarks being drawn
    cv::imshow("Input image", Image);
    cv::imshow("Output image", inputImage);

    //It will pause the program until you press any key from keyboard So that you can see output.
    cv:: waitKey(0);

    return 0;
}
