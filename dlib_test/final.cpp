#include<iostream>
#include<string>
#include<dlib/opencv.h>
#include<dlib/image_io.h>
#include<dlib/image_processing.h>
#include<dlib/image_processing/frontal_face_detector.h>
#include<dlib/image_processing/render_face_detections.h>
#include<opencv4/opencv2/highgui.hpp>
#include<opencv4/opencv2/opencv.hpp>
#include<opencv4/opencv2/highgui/highgui.hpp>
#include<opencv4/opencv2/core/core.hpp>
#include<opencv4/opencv2/imgproc.hpp> 

using namespace dlib;
using namespace std;
using namespace cv;


//draw landmarks over face
void drawFaceLandmarks(Mat &image, full_object_detection faceLandmark){

    //Loop over all face landmarks
    for(int i=0; i< faceLandmark.num_parts(); i++){
        int x = faceLandmark.part(i).x();
        int y = faceLandmark.part(i).y();
        string text = to_string(i+1);

        //Draw a small circle at face landmark over the image using opencv
        circle(image, Point(x, y), 1, Scalar(0, 0, 255), 2, LINE_AA );

        //Draw text at face landmark to show index of current face landmark over the image using opencv
        putText(image, text, Point(x, y), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 0, 0), 1);
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

int main(){

    int eye_centre_x_left, eye_centre_y_left,eye_centre_x_right, eye_centre_y_right;    
    image_window win, win_faces;
    //Load the dlib face detector
    frontal_face_detector faceDetector = get_frontal_face_detector();

    //Load the dlib face landmark detector
    shape_predictor faceLandmarkDetector ;
    deserialize("/home/varunsakunia/Downloads/shape_predictor_5_face_landmarks.dat") >> faceLandmarkDetector;
    //deserialize("/home/varunsakunia/shape_predictor_68_face_landmarks.dat") >> faceLandmarkDetector;

    //And finally we load the DNN responsible for face recognition
    //std::string face_rec_path = "/home/varunsakunia/dlib_face_recognition_resnet_model_v1.dat";
    //face_recognition_model_v1 face_encoder = face_recognition_model_v1(face_rec_path);

    // Read image using opencv
    Mat inputImage = imread("/home/varunsakunia/Desktop/id011.jpg");

    //Check if the mnetioned image exits
    if(inputImage.empty()){
        cout<<"image doesn't exist"<<endl;
        return -1;
    }

    //create a copy of the input image to work on so that finally we have input and output images to compare
    Mat image = inputImage.clone();

    //Convert loaded opencv image to dlib image format
    cv_image<bgr_pixel> dlibImage(image);

    //Detect faces in the image and print the number of faces detected
    std::vector<dlib::rectangle> faces = faceDetector(dlibImage);
    cout<<"Number of faces detected:"<<faces.size()<<endl;
   
    // Now tell the face detector to give us a list of bounding boxes around all the faces in the image.
    std::vector<dlib::rectangle> dets = faceDetector(dlibImage);
    cout << "Number of faces detected: " << dets.size() << endl;

    // Now we will go ask the shape_predictor to tell us the pose of each face we detected.
    std::vector<dlib::full_object_detection> shapes;
            for (unsigned long j = 0; j < dets.size(); ++j)
            {
                full_object_detection shape = faceLandmarkDetector(dlibImage, dets[j]);
                cout << "Number of parts detected by dlib: "<< shape.num_parts() << endl;
               //cout << "pixel position of first part: " << shape.part(0) << endl;
               // cout << "pixel position of second part: " << shape.part(1) << endl;
	        eye_centre_x_left = ((shape.part(2) + shape.part(3))/2).x();
		eye_centre_y_left = ((shape.part(2) + shape.part(3))/2).y();
 		eye_centre_x_right = ((shape.part(0) + shape.part(1))/2).x();
                eye_centre_y_right = ((shape.part(0) + shape.part(1))/2).y();		
		cout << "Eye Centres (L-R): " << (shape.part(2) + shape.part(3))/2 <<" "<< (shape.part(0) + shape.part(1))/2 <<endl;
		//cv::circle(dlibImage, (shape.part(2) + shape.part(3))/2,5, Scalar(0,0,255), 3);
		//cv::circle(dlibImage, (shape.part(0) + shape.part(1))/2,5, Scalar(0,0,255), 3);
	
		// You get the idea, you can get all the face part locations if
                // you want them.  Here we just store them in shapes so we can
                // put them on the screen.
                shapes.push_back(shape);
            }

     // Now let's view our face poses on the screen.
     win.clear_overlay();
     win.set_image(dlibImage);
     win.add_overlay(render_face_detections(shapes));

    // We can also extract copies of each face that are cropped, rotated upright, and scaled to a standard size as shown here:
    dlib::array<array2d<rgb_pixel> > face_chips;
    extract_image_chips(dlibImage, get_face_chip_details(shapes), face_chips);
    win_faces.set_image(tile_images(face_chips));

    //saving face chip
    int image_id = 1;
    for (int idx = 0; idx < face_chips.size(); idx++){
                std::string fname = "detected_facechip" + std::to_string(image_id) + ".jpg";
                dlib::save_jpeg(face_chips[idx], fname);
                image_id++;
            }
    
    //Printing the x and y corrdinates of eye centres (L-R)
    cout << "Centre Coordinates: " << eye_centre_x_left << " " << eye_centre_y_left << " " << eye_centre_x_right << " " << eye_centre_y_right << endl;
   
    //Get Face landmarks of all detected faces
    std::vector<full_object_detection> facelandmarks;
    for(int i=0; i<faces.size(); i++){

        //Get the face landmark and print number of landmarks detected
        full_object_detection facelandmark = faceLandmarkDetector(dlibImage, faces[i]);
        cout<<"Number of face landmarks detected:"<<facelandmark.num_parts()<<endl;

        //Push face landmark to array of All face's landmarks array
        facelandmarks.push_back(facelandmark);

        //Draw landmarks on image
        drawFaceLandmarks(image, facelandmark);

        //Write face Landmarks to a file on disk to analyze
        string landmarksFilename = "face_landmarks_" + to_string(i+1) + ".txt";
        writeFaceLandmarkstoFile(landmarksFilename, facelandmark);
    }

    //Save image with face landmarks drawn to disk to analyze using opencv
    string imageFileName = "image_face_landmark.jpg";
    cv::imwrite(imageFileName,image);

    //####################(  Draw Eye Centres  )########################
    cv::Point centerCircle_left(eye_centre_x_left, eye_centre_y_left);
    int radiusCircle = 3;
    cv::Scalar colorCircle(0,0,255);
    int thicknessCircle = 2;
    cv::circle(image, centerCircle_left, radiusCircle, colorCircle, thicknessCircle);
    cv::Point centerCircle_right(eye_centre_x_right, eye_centre_y_right);
    cv::circle(image, centerCircle_right, radiusCircle, colorCircle, thicknessCircle);
 
    //Create windows to show images
    namedWindow("Input image", WINDOW_NORMAL);
    namedWindow("Output image", WINDOW_NORMAL);

    //Display both images with face landmarks and without face landmarks being drawn
    imshow("Input image", inputImage);
    imshow("Output image", image);

    //It will pause the program until you press any key from keyboard So that you can see output.
    waitKey(0);

    return 0;
}
