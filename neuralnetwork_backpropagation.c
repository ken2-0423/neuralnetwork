/*----------------------------------------------------*/
/*   Program name : neuralnetwork_backpropagation.c   */
/*    Date of program : 2024/9/24                     */
/*   Author : Ueno                                   */
/*----------------------------------------------------*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define INPUTNO 3 /*入力層のニューロン数*/
#define HIDDENNO 3/*中間層のニューロン数*/
#define OUTPUTNO 2/*出力層のニューロン数(教師データの数も同じになる)*/
// ALPHA（学習係数）を可変にするための設定
#define ALPHA 0.5  //学習係数
#define SEED 120/*乱数のシード*/
#define MAXINPUTNO 8 /*学習データの最大個数*/
#define MAXOUTPUTNO 8/*学習データの最大個数*/
#define BIGNUM 100/*誤差の初期値*/
#define LIMIT 0.01/*誤差の上限値*/
// MAXEPOCH（エポック数）を追加して、明示的にループ回数を制御
#define MAXEPOCH 1000  // エポック数の最大値を1000に設定

/*プロトタイプ宣言*/
void initwh(double wh[HIDDENNO] [INPUTNO + 1]);/*中間層の重みの初期化*/
void initwy(double wy[OUTPUTNO] [HIDDENNO + 1]);/*出力層の重みの初期化*/
double f(double u);/*伝達関数*/
double diff_f(double u);
double drnd(void);/*乱数の生成*/
void forward(double wh[HIDDENNO] [INPUTNO + 1], double wy[OUTPUTNO] [HIDDENNO + 1], double hi[HIDDENNO], double x[INPUTNO + OUTPUTNO], double y[OUTPUTNO]);/*順方向の計算*/
void ylearn(double wy[OUTPUTNO] [HIDDENNO + 1], double hi [HIDDENNO], double x[INPUTNO + OUTPUTNO], double y[OUTPUTNO], double y_d[OUTPUTNO]);/*出力層の学習*/
void ylearn_update(double wy[OUTPUTNO] [HIDDENNO + 1], double hi [HIDDENNO], double x[INPUTNO + OUTPUTNO], double y[OUTPUTNO], double y_d[OUTPUTNO], double wy_d[OUTPUTNO] [HIDDENNO + 1]);/*出力層の重みの更新*/
void hlearn(double wh[HIDDENNO] [INPUTNO + 1], double wy [OUTPUTNO] [HIDDENNO + 1], double hi[HIDDENNO], double x[INPUTNO + OUTPUTNO], double y[OUTPUTNO], double hi_d[HIDDENNO]);/*中間層の学習*/
void hlearn_update(double wh[HIDDENNO] [INPUTNO + 1], double wy [OUTPUTNO] [HIDDENNO + 1], double hi[HIDDENNO], double x[INPUTNO + OUTPUTNO], double y[OUTPUTNO], double hi_d[HIDDENNO], double wh_d[HIDDENNO] [INPUTNO + 1]);/*中間層の重みの更新*/
void print_xhy(double x[MAXINPUTNO] [INPUTNO + OUTPUTNO], double hi[HIDDENNO], double y[MAXOUTPUTNO] [OUTPUTNO], double y_d[OUTPUTNO], double hi_d[HIDDENNO], double wh_d[HIDDENNO] [INPUTNO + 1], double wy_d[OUTPUTNO] [HIDDENNO + 1]);/*途中経過の出力*/
void print_w(double wh[HIDDENNO] [INPUTNO + 1], double wy[OUTPUTNO] [HIDDENNO + 1]);/*結果の出力*/
int getdata(double x[] [INPUTNO + OUTPUTNO]);/*データの読み込み*/
void wait_enter(void);/*Enter押下検知*/

/*main()関数*/
int main()
{
    double wh[HIDDENNO] [INPUTNO + 1];/*中間層の重み*/
    double wy[OUTPUTNO] [HIDDENNO + 1];/*出力層の重み*/
    double x[MAXINPUTNO] [INPUTNO + OUTPUTNO];/*学習データセット*/
    double hi[HIDDENNO];/*中間層の出力*/
    double y[MAXOUTPUTNO] [OUTPUTNO];/*出力*/
    double y_d[OUTPUTNO];/*出力層の重み計算に利用*/
    double hi_d[HIDDENNO];/*中間層の重み計算に利用*/
    double wy_d[OUTPUTNO] [HIDDENNO + 1];/*出力層の重み勾配*/
    double wh_d[HIDDENNO] [INPUTNO + 1];/*中間層の重み勾配*/
    double err = BIGNUM;/*誤差の評価*/
    int i, j, k;/*繰り返しの制御*/
    int n_of_x;/*データの個数*/
    int count = 0;/*繰り返し回数のカウンタ*/

    FILE *count_err;/*学習回数と誤差の結果をCSVファイルに出力用*/
    count_err = fopen("count_err.csv", "w");
    if(count_err == NULL){
        printf("File not opened...\n");
        return 1;
    }

    /*乱数の初期化*/
    srand(SEED);

    /*重みの初期化*/
    initwh(wh);
    initwy(wy);
    print_w(wh, wy);

    /*学習データの読み込み。ただし、＋OUTPUNTO個の部位分は教師データとなる*/
    n_of_x = getdata(x);
    printf("データの個数 : %d\n", n_of_x);

    /*学習*/
    while (err > LIMIT){
        err = 0.0;
        for(i = 0; i < n_of_x; i++){
            /*順方向の計算*/
            forward(wh, wy, hi, x[i], y[i]);

            /*出力層の重み更新*/
            ylearn(wy, hi, x[i], y[i], y_d);

            /*中間層の重み更新*/
            hlearn(wh, wy, hi, x[i], y[i], hi_d);
            /*重みデータの保存*/

            /*誤差の計算*/
            for(j = 0; j < OUTPUTNO; j++){
                err += ((x[i] [INPUTNO + j] - y[i] [j]) * (x[i] [INPUTNO + j] - y[i] [j])) / 2;
            }

            ylearn_update(wy, hi, x[i], y[i], y_d, wy_d);
            hlearn_update(wh, wy, hi, x[i], y[i], hi_d, wh_d);

            /*入力、出力、中間層の出力*/
            //print_xhy(x, hi, y, y_d, hi_d, wh_d, wy_d);
            /*重みの出力*/
            //print_w(wh, wy);
        }

        count++;

        /*コマンドラインに誤差の出力*/
        printf("roop:%d\t誤差:%lf\n", count, err);
        /*CSVファイルに誤差の出力*/
        fprintf(count_err, "%d,%lf\n", count, err);
    }

    fclose(count_err);

    /*予測*/
    double x_test[INPUTNO + OUTPUTNO] = {0,1,0,0,1};

    double y_test[OUTPUTNO];/*x_testの予測結果を格納*/

    forward(wh, wy, hi, x_test, y_test);
    printf("予測の出力結果：\n");
    for(i = 0; i < OUTPUTNO; i++){
        printf("%f ",y_test[i]);/*予測結果を表示*/
    }
    printf("\n");

    return 0;
}

/*wait_enter()関数*/
void wait_enter(void)
{
    int c;
    while ((c = getchar()) != '\n'){
        if (c == EOF) {
            puts("Error out!");
            exit(1);
        }
    }
}

/*getdata()関数*/
int getdata(double x[] [INPUTNO + OUTPUTNO])
{
    int n_of_x = 0;/*データセットの個数*/
    int j = 0;/*繰り返しの制御*/
    FILE *file;/*ファイル読み込み用*/

    file = fopen("neuralnetwork_backpropagation_dataset.txt","r");
    if(file == NULL){
        printf("File not found...\n");
        return 1;
    }

    /*データの入力*/
    while(fscanf(file, "%lf", &x[n_of_x][j]) != EOF){
        j++;
        if(j >= INPUTNO + OUTPUTNO){
            j = 0;
            n_of_x++;
        }
    }

    fclose(file);

    return n_of_x;
}

/*print_w()関数*/
void print_w(double wh[HIDDENNO] [INPUTNO + 1], double wy[OUTPUTNO] [HIDDENNO + 1])
{
    int i, j;

    printf("入力層-中間層の重み\n");
    for(i = 0; i < HIDDENNO; i++){
        for(j = 0; j < INPUTNO + 1; j++){
            printf("wh[%d] [%d] = %f\t", i, j, wh[i] [j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("中間層-出力層の重み\n");
    for(i = 0; i < OUTPUTNO; i++){
        for(j = 0; j < HIDDENNO + 1; j++){
            printf("wy[%d] [%d] = %f\t", i, j, wy[i] [j]);
        }
        printf("\n");
    }

    wait_enter();
    printf("\n");
}

/*print_xhy()関数*/
void print_xhy(double x[MAXINPUTNO] [INPUTNO + OUTPUTNO], double hi[HIDDENNO], double y[MAXOUTPUTNO] [OUTPUTNO], double y_d[OUTPUTNO], double hi_d[HIDDENNO], double wh_d[HIDDENNO] [INPUTNO + 1], double wy_d[OUTPUTNO] [HIDDENNO + 1])
{
    int i, j;/*繰り返しの制御*/

    printf("入力:\n");
    for(i = 0; i < MAXINPUTNO; i++){
        for(j = 0; j < INPUTNO; j++){
            printf("x[%d] [%d] = %f\t", i, j, x[i] [j]);
        }
        printf("\n");
    }

    printf("中間出力:\n");
    for(i = 0; i < HIDDENNO; i++){
        printf("hi[%d] = %f\t", i, hi[i]);
    }
    printf("\n");

    printf("出力:\n");
    for(i = 0; i < MAXOUTPUTNO; i++){
        for(j = 0; j < OUTPUTNO; j++){
            printf("y[%d] [%d] = %f\t", i, j, y[i] [j]);
        }
        printf("\n");
    }

    printf("出力誤差:\n");
    for(i = 0; i < OUTPUTNO; i++){
        printf("y_d[%d] = %f\t", i, y_d[i]);
    }
    printf("\n");

    printf("中間層-出力層間の重み勾配\n");
    for(i = 0; i < OUTPUTNO; i++){
        for(j = 0; j < HIDDENNO + 1; j++){
            printf("wy_d[%d] [%d] = %f\t", i, j, wy_d[i] [j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("中間層出力誤差:\n");
    for(i = 0; i < HIDDENNO; i++){
        printf("hi_d[%d] = %f\t", i, hi_d[i]);
    }
    printf("\n");

    printf("入力層-中間層間の重み勾配\n");
    for(i = 0; i < HIDDENNO; i++){
        for(j = 0; j < INPUTNO + 1; j++){
            printf("wh_d[%d] [%d] = %f\t", i, j, wh_d[i] [j]);
        }
        printf("\n");
    }
    printf("\n");
}

/*hlearn()関数*/
void hlearn(double wh[HIDDENNO] [INPUTNO + 1], double wy[OUTPUTNO] [HIDDENNO + 1], double hi[HIDDENNO], double x[INPUTNO + OUTPUTNO], double y[OUTPUTNO], double hi_d[HIDDENNO])
{
    int i, j;/*繰り返しの制御*/

    for(i = 0; i < HIDDENNO; i++){
        for(j = 0; j < OUTPUTNO; j++){
            hi_d[i] += ((y[j] - x[INPUTNO + j]) * diff_f(y[j])) * wy[j] [i];
        }
        hi_d[i] = hi_d[i] * diff_f(hi[i]);
    }
}

void hlearn_update(double wh[HIDDENNO] [INPUTNO + 1], double wy[OUTPUTNO] [HIDDENNO + 1], double hi[HIDDENNO], double x[INPUTNO + OUTPUTNO], double y[OUTPUTNO], double hi_d[HIDDENNO], double wh_d[HIDDENNO] [INPUTNO + 1])
{
    int i, j;/*繰り返しの制御*/

    for(i = 0; i < HIDDENNO; i++){
        for(j = 0; j < INPUTNO; j++){
            wh_d[i] [j] = x[j] * hi_d[i];
            wh[i] [j] -= ALPHA * wh_d[i] [j];
        }
        wh_d[i] [j] = hi_d[i];
        wh[i] [j] += ALPHA * wh_d[i] [j];/*閾値の学習*/
    }
}

/*ylearn()関数*/
void ylearn(double wy[OUTPUTNO] [HIDDENNO + 1], double hi[HIDDENNO], double x[INPUTNO + OUTPUTNO], double y[OUTPUTNO], double y_d[OUTPUTNO])
{
    int i;/*繰り返しの制御*/

    for(i = 0; i < OUTPUTNO; i++){
       y_d[i] = (y[i] - x[INPUTNO + i]) * diff_f(y[i]);/*誤差の計算*/
    }
}

void ylearn_update(double wy[OUTPUTNO] [HIDDENNO + 1], double hi[HIDDENNO], double x[INPUTNO + OUTPUTNO], double y[OUTPUTNO], double y_d[OUTPUTNO], double wy_d[OUTPUTNO] [HIDDENNO + 1])
{
    int i, j;/*繰り返しの制御*/

    for(i = 0; i < OUTPUTNO; i++){
        for(j = 0; j < HIDDENNO; j++){
            wy_d[i] [j] = hi[j] * y_d[i];
            wy[i] [j] -= ALPHA * wy_d[i] [j];/*重みの学習*/
        }
        wy_d[i] [j] = y_d[i];
        wy[i] [j] += ALPHA * wy_d[i] [j];/*閾値の学習*/
    }
}

/*forward()関数*/
void forward(double wh[HIDDENNO] [INPUTNO + 1], double wy[OUTPUTNO] [HIDDENNO + 1], double hi[HIDDENNO], double x[INPUTNO + OUTPUTNO], double y[OUTPUTNO])
{
    int i, j;/*繰り返しの制御*/
    double u;/*重み付き和u*/

    /*hiの計算*/
    for(i = 0; i < HIDDENNO; i++){
        u = 0;
        /*重み付き和の計算*/
        for(j = 0; j < INPUTNO; j++){
            u += x[j] * wh[i] [j];
        }
        u -= wh[i] [j];/*閾値の処理*/
        /*隠れ層出力値の計算*/
        hi[i] = f(u);/*伝達関数の計算*/
    }    

    /*出力yの計算*/
    for(i = 0; i < OUTPUTNO; i++){
        y[i] = 0;
        for(j = 0; j < HIDDENNO + 1; j++){
            y[i] += hi[j] * wy[i] [j];
        }
        y[i] -= wy[i] [j];/*閾値の処理*/

        y[i] = f(y[i]);/*活性化関数の計算*/
    }
}

/*drnd()関数*/
double drnd(void)
{
    double rndno;/*生成した乱数*/

    while((rndno = (double)rand() / RAND_MAX) == 1.0);
    rndno = rndno * 2 - 1;/*-1から1の間の乱数を生成*/

    return rndno;
}

/*f()関数*/
double f(double u)
{
    /*シグモイド関数*/
    return 1.0 / (1.0 + exp(-1.0 * u));
}

/*f()関数の微分*/
double diff_f(double u)
{
    /*シグモイド関数*/
    return u * (1.0 - u);
}

/*initwh()関数*/
void initwh(double wh[HIDDENNO] [INPUTNO + 1])
{
    int i, j;/*繰り返しの制御*/

    /*乱数による重みの決定*/
    for(i = 0; i < HIDDENNO; i++){
        for(j = 0; j < INPUTNO + 1; j++){
            wh[i] [j] = drnd();
        }
    }
}

/*initwy()関数*/
void initwy(double wy[OUTPUTNO] [HIDDENNO + 1])
{
    int i, j;/*繰り返しの制御*/

    /*乱数による重みの決定*/
    for(i = 0; i < OUTPUTNO; i++){
        for(j = 0; j < HIDDENNO + 1; j++){
            wy[i] [j] = drnd();
        }
    }
}
