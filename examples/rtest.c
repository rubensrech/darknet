#include "darknet.h"

// > Auxiliary functions

void print_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes) {
    int i,j;

    for(i = 0; i < num; ++i){
        char labelstr[4096] = {0};
        int _class = -1;
        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j] > thresh){
                if (_class < 0) {
                    strcat(labelstr, names[j]);
                    _class = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
            }
        }
    }
}

// > Tests

/* 
 * Test 1
 * Description: Test average execution time for one frame after 'n' predictions (default n=1)
 * Call: ./darknet detector rtest 1 <cfgfile> <weightfile> <filename> [-n <iterations>] [-print <0|1>]
 * Optionals: -n <iterations>   - default: 1
 *            -print <0|1>      - print detections? default: 1
 */
void test1(char *cfgfile, char *weightfile, char *filename, int n, int print) {
    char *datacfg = (char*)"cfg/coco.data";
    float thresh = 0.3;
    float hier_thresh = 0.5;

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, (char*)"names", (char *)"data/names.list");
    char **names = get_labels(name_list);
    image **alphabet = load_alphabet();

    double tl = what_time_is_it_now();

    network *net = load_network(cfgfile, weightfile, 0);

    printf("Load net time: %f ms.\n\n", (what_time_is_it_now() - tl) * 1000);

    set_batch_network(net, 1);
    srand(2222222);

    char buff[256];
    char *input = buff;
    float nms = .45;

    strncpy(input, filename, 256);

    layer l = net->layers[net->n - 1];

    image im, sized;
    int nboxes = 0;
    detection *dets;

    double t = what_time_is_it_now();

    int i;
    for (i = 0; i < n; i++) {
        // Load input image
        im = load_image_color(input, 0, 0);
        sized = letterbox_image(im, net->w, net->h);

        float *X = sized.data;

        // Run predictor
        network_predict(net, X);

        // Generate outputs
        int letterbox = 1;
        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);

        if (nms)
            do_nms_sort(dets, nboxes, l.classes, nms);

        if (print) {
            print_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
            printf("\n");
        } else {
            fprintf(stderr, "\r%d ", i+1);
        }

        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
    }

    double tt = (what_time_is_it_now() - t) * 1000;
    printf("\nTotal time: %f ms.\n", tt);
    if (n > 1)
        printf("Average time: %f ms.\n", tt/n);

    free_network(net);
}

/* 
 * Test 2
 * Description: Test average execution time for one frame with 'n' COCO validation images
 * Details: First prediction time is discarded because of network push cost
 * Call: ./darknet detector rtest 2 <cfgfile> <weightfile> [-n <images>]
 * Optionals: -n <images> - number of images from COCO dataset (max: 4999, min: 2, default: 5000)
 */
void test2(char *cfgfile, char *weightfile, int n) {
    network *net = load_network(cfgfile, weightfile, 0);
    layer l = net->layers[net->n-1];

    char *valid_images = (char*)"../coco_test/5k.txt";
    list *plist = get_paths(valid_images);
    char **paths = (char**)list_to_array(plist);
    if (n > plist->size) {
        fprintf(stderr, "Argument 'n' cannot be greater than 5000!");
        return;
    }

    set_batch_network(net, 1);
    srand(22222);

    image im, sized;
    int nboxes = 0;
    detection *dets;

    float nms = .45;
    float thresh = 0.3;
    float hier_thresh = 0.5;

    double t1 = 0;

    int i;
    for (i = 0; i < n; i++) {
        if (i == 1) // Discard first frame (because of network push cost)
            t1 = what_time_is_it_now();

        im = load_image_color(paths[i], 0, 0);
        sized = letterbox_image(im, net->w, net->h);
        float *X = sized.data;

        network_predict(net, X);

        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, 1);
        if (nms)
            do_nms_sort(dets, nboxes, l.classes, nms);
        // print_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);

        free_image(im);
        free_image(sized);
        free_detections(dets, nboxes);

        fprintf(stderr, "\r%d ", i+1);
    }

    double t = what_time_is_it_now() - t1;
    printf("\nTotal time for %d frame(s): %f s\n", n, t);
    printf("Time per frame: %f ms\n", t/n * 1000);

    free_network(net);
}

/* 
 * Test 3
 * Description: Test average execution time for each layer after 'n' predictions
 * Call: ./darknet detector rtest 3 <cfgfile> <weightfile> <filename> [-n <iterations>]
 * Details: First prediction time is discarded because of network push cost
 *          Important: set LAYERS_TIME_TEST to 1 in network.c!
 * Optionals: -n <iterations>   - default: 50
 */

/* Format: layers_times[layer][iteration] */
float **layers_times;

void test3(char *cfgfile, char *weightfile, char *filename, int n) {
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    char buff[256];
    char *input = buff;
    strncpy(input, filename, 256);

    image im = load_image_color(input, 0, 0);
    image sized = letterbox_image(im, net->w, net->h);
    float *X = sized.data;

    // Alloc times matrix
    int j;
    layers_times = (float**)malloc(net->n * sizeof(float*));
    for (j = 0; j < net->n; j++)
        layers_times[j] = (float*)malloc(n * sizeof(float));

    int i;
    for (i = 0; i < n; i++) {
        network_predict(net, X);
        fprintf(stderr, "\r%d ", i+1);
    }

    // Calculate average (discarding first iteration of each layer)
    printf("\n");
    for (i = 0; i < net->n; i++) {
        double sum = 0;
        for (j = 1; j < n; j++) sum += layers_times[i][j];
        double avg = sum/(n-1);
        printf("layer %3d: %f ms\n", i, avg*1000);
    }

    free_image(im);
    free_image(sized);
    free_network(net);    
}

void run_rtest(int testID, int argc, char **argv) {
    if (testID == 1) {
        char *cfgfile = argv[4];
        char *weightfile = argv[5];
        char *filename = argv[6];
        int n = find_int_arg(argc, argv, (char*)"-n", 1);
        int print = find_int_arg(argc, argv, (char*)"-print", 1);
        test1(cfgfile, weightfile, filename, n, print);
    } else if (testID == 2) {
        char *cfgfile = argv[4];
        char *weightfile = argv[5];
        int n = find_int_arg(argc, argv, (char*)"-n", 1);
        test2(cfgfile, weightfile, n);
    } else if (testID == 3) {
        char *cfgfile = argv[4];
        char *weightfile = argv[5];
        char *filename = argv[6];
        int n = find_int_arg(argc, argv, (char*)"-n", 50);
        test3(cfgfile, weightfile, filename, n);
    } else {
        printf("Invalid test ID!\n");
    }
}