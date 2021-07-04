# fencing_yolo

Create a folder in the root directory called "extracted_data". Inside this folder, create folders called "raw_box_list_lists", "true_left_right_csvs", 
"right_lists", "left_lists", "true_right_lists", "true_left_lists", "disp_arrs", "box_list_lists".

Save a clip, and create a Clip object using a = Clip(filepath, svm), where svm is the support vector machine loaded from svm.pt.

Then run a.main() to compute all the data needed, and finally use a.save_data('extracted_data', filename), where filename is the desired name for the files.
