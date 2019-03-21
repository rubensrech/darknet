#include "darknet.h"

void fix_data_captcha(data d, int mask)
{
    matrix labels = d.y;
    int i, j;
    for(i = 0; i < d.y.rows; ++i){
        for(j = 0; j < d.y.cols; j += 2){
            if (mask){
                if(!labels.vals[i][j]){
                    labels.vals[i][j] = SECRET_NUM;
                    labels.vals[i][j+1] = SECRET_NUM;
                }else if(labels.vals[i][j+1]){
                    labels.vals[i][j] = 0;
                }
            } else{
                if (labels.vals[i][j]) {
                    labels.vals[i][j+1] = 0;
                } else {
                    labels.vals[i][j+1] = 1;
                }
            }
        }
    }
}

void train_captcha(char *cfgfile, char *weightfile)
{
    srand(time(0));
    real avg_loss = CAST(-1);
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network *net = load_network(cfgfile, weightfile, 0);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", (float)net->learning_rate, (float)net->momentum, (float)net->decay);
    int imgs = 1024;
    int i = *net->seen/imgs;
    int solved = 1;
    list *plist;
    char **labels = get_labels((char*)"/data/captcha/reimgs.labels.list");
    if (solved){
        plist = get_paths((char*)"/data/captcha/reimgs.solved.list");
    }else{
        plist = get_paths((char*)"/data/captcha/reimgs.raw.list");
    }
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    pthread_t load_thread;
    data train;
    data buffer;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.classes = 26;
    args.n = imgs;
    args.m = plist->size;
    args.labels = labels;
    args.d = &buffer;
    args.type = CLASSIFICATION_DATA;

    load_thread = load_data_in_thread(args);
    while(1){
        ++i;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        fix_data_captcha(train, solved);


        load_thread = load_data_in_thread(args);
        printf("Loaded: %f seconds\n", (float)sec(clock()-time));
        time=clock();
        real loss = train_network(net, train);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %f seconds, %ld images\n", i, (float)loss, (float)avg_loss, (float)sec(clock()-time), *net->seen);
        free_data(train);
        if(i%100==0){
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/%s_%d.weights",base, i);
            save_weights(net, buff);
        }
    }
}

void test_captcha(char *cfgfile, char *weightfile, char *filename)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    int i = 0;
    char **names = get_labels((char*)"/data/captcha/reimgs.labels.list");
    char buff[256];
    char *input = buff;
    int indexes[26];
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            //printf("Enter Image Path: ");
            //fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, net->w, net->h);
        float *X = im.data;
        real *predictions = network_predict_float(net, X);
        top_predictions(net, 26, indexes);
        //printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < 26; ++i){
            int index = indexes[i];
            if(i != 0) printf(", ");
            printf("%s %f", names[index], (float)predictions[index]);
        }
        printf("\n");
        fflush(stdout);
        free_image(im);
        if (filename) break;
    }
}

void valid_captcha(char *cfgfile, char *weightfile, char *filename)
{
    char **labels = get_labels((char*)"/data/captcha/reimgs.labels.list");
    network *net = load_network(cfgfile, weightfile, 0);
    list *plist = get_paths((char*)"/data/captcha/reimgs.fg.list");
    char **paths = (char **)list_to_array(plist);
    int N = plist->size;
    int outputs = net->outputs;

    set_batch_network(net, 1);
    srand(2222222);
    int i, j;
    for(i = 0; i < N; ++i){
        if (i%100 == 0) fprintf(stderr, "%d\n", i);
        image im = load_image_color(paths[i], net->w, net->h);
        float *X = im.data;
        real *predictions = network_predict_float(net, X);
        //printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        int truth = -1;
        for(j = 0; j < 13; ++j){
            if (strstr(paths[i], labels[j])) truth = j;
        }
        if (truth == -1){
            fprintf(stderr, "bad: %s\n", paths[i]);
            return;
        }
        printf("%d, ", truth);
        for(j = 0; j < outputs; ++j){
            if (j != 0) printf(", ");
            printf("%f", (float)predictions[j]);
        }
        printf("\n");
        fflush(stdout);
        free_image(im);
        if (filename) break;
    }
}

void run_captcha(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "train")) train_captcha(cfg, weights);
    else if(0==strcmp(argv[2], "test")) test_captcha(cfg, weights, filename);
    else if(0==strcmp(argv[2], "valid")) valid_captcha(cfg, weights, filename);
}

