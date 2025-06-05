using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace AIMouseTrainer
{
    public partial class PredictorForm : Form
    {
        // burdan eğitimi aç kapa: şuan açık
        // Eğitim açıkken mouse'u hedefe götür ve bunu istediğin kadar tekrar et,
        // Eğitimi bitirmek istediğinde S tuşuna bas ve model eğitimi başlasın. Bitene kadar programı kapatma.
        // Eğitim tamamlandıktan sonra *TRAINING_MODE = false* olarak ayarla, programı yeniden aç ve V tuşuna bas.
        // Artık AI mouse hareketlerini taklit etmeye başlayacak rastgele hedeflere doğru.
        const bool TRAINING_MODE = true; 


        const int WIDTH = 800, HEIGHT = 600;
        const int FPS = 200;
        const int TARGET_RADIUS = 15;
        const double MOUSE_STEP_SIZE = 4.0;

        const int MOUSE_HISTORY_SEQUENCE_LENGTH = 10;
        const int HISTORY_POINTS_NEEDED_FOR_FEATURES = 4;
        const int FEATURES_PER_TIMESTEP = 9;
        const int TOTAL_INPUT_FEATURES = MOUSE_HISTORY_SEQUENCE_LENGTH * FEATURES_PER_TIMESTEP;
        const int MAX_POINTS = MOUSE_HISTORY_SEQUENCE_LENGTH + HISTORY_POINTS_NEEDED_FOR_FEATURES + 50;

        const double MAX_SPEED_INPUT_NORMALIZATION = MOUSE_STEP_SIZE * 3.0;
        const double MIN_SPEED_MULTIPLIER = 0.2;
        const double MAX_SPEED_MULTIPLIER = 3.0;
        const double MAX_ACCELERATION_NORMALIZATION = 2.0 * MOUSE_STEP_SIZE;
        const double ANGULAR_CHANGE_NORMALIZATION_DIVISOR = Math.PI;


        abstract class Layer
        {
            public abstract double[] Forward(double[] input);
            public abstract double[] Backpropagate(double[] error, double lr);
            public abstract object Save();
            public abstract void Load(JsonElement data);
        }

        class DenseLayer : Layer
        {
            readonly int inSize, outSize;
            public string activation;
            readonly double[,] W;
            readonly double[] B;
            double[] lastInput, lastOutput;
            static readonly Random rng = new Random();

            public DenseLayer(int input, int output, string act)
            {
                inSize = input; outSize = output; activation = act;
                W = new double[inSize, outSize];
                B = new double[outSize];
                double scale = Math.Sqrt(6.0 / (inSize + outSize));
                for (int i = 0; i < inSize; i++)
                    for (int j = 0; j < outSize; j++)
                        W[i, j] = (rng.NextDouble() * 2 - 1) * scale;
            }

            public override double[] Forward(double[] input)
            {
                if (input.Length != inSize)
                {
                    Console.WriteLine($"HATA: DenseLayer.Forward - Beklenen giriş boyutu {inSize}, gelen {input.Length}");
                    return new double[outSize];
                }
                lastInput = input;
                double[] z = new double[outSize];
                for (int j = 0; j < outSize; j++)
                {
                    double sum = B[j];
                    for (int i = 0; i < inSize; i++) sum += input[i] * W[i, j];
                    z[j] = sum;
                }
                if (activation == "relu")
                    for (int j = 0; j < outSize; j++) z[j] = Math.Max(0, z[j]);
                else if (activation == "tanh")
                    for (int j = 0; j < outSize; j++) z[j] = Math.Tanh(z[j]);
                lastOutput = z;
                return z;
            }

            public override double[] Backpropagate(double[] error, double lr)
            {
                if (lastInput == null || lastOutput == null)
                {
                    Console.WriteLine("HATA: DenseLayer.Backpropagate - Forward çağrılmadan Backprop denendi veya giriş/çıkış eksik.");
                    return new double[inSize];
                }
                if (error.Length != outSize)
                {
                    Console.WriteLine($"HATA: DenseLayer.Backpropagate - Beklenen hata boyutu {outSize}, gelen {error.Length}.");
                    return new double[inSize];
                }

                double[] gradAct = new double[outSize];
                for (int j = 0; j < outSize; j++)
                    gradAct[j] = activation == "relu"
                        ? (lastOutput[j] > 0 ? error[j] : 0.0)
                        : error[j] * (1 - lastOutput[j] * lastOutput[j]);

                for (int i = 0; i < inSize; i++)
                    for (int j = 0; j < outSize; j++)
                        W[i, j] -= lr * lastInput[i] * gradAct[j];
                for (int j = 0; j < outSize; j++) B[j] -= lr * gradAct[j];

                double[] errorPrev = new double[inSize];
                for (int i = 0; i < inSize; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < outSize; j++) sum += gradAct[j] * W[i, j];
                    errorPrev[i] = sum;
                }
                return errorPrev;
            }
            public override object Save()
            {
                var wJagged = new double[inSize][];
                for (int i = 0; i < inSize; i++)
                {
                    wJagged[i] = new double[outSize];
                    for (int j = 0; j < outSize; j++) wJagged[i][j] = W[i, j];
                }
                return new { type = "DenseLayer", inSize, outSize, activation, W = wJagged, B };
            }
            public override void Load(JsonElement data)
            {
                activation = data.GetProperty("activation").GetString();
                var rowEnum = data.GetProperty("W").EnumerateArray();
                int i = 0;
                foreach (var rowElem in rowEnum)
                {
                    int j = 0;
                    foreach (var val in rowElem.EnumerateArray()) { if (i < inSize && j < outSize) W[i, j] = val.GetDouble(); j++; }
                    i++;
                }
                var bArr = data.GetProperty("B").EnumerateArray().ToArray();
                for (int j = 0; j < outSize && j < bArr.Length; j++) B[j] = bArr[j].GetDouble();
            }
        }

        class DropoutLayer : Layer
        {
            readonly double rate;
            bool[] mask;
            bool isTraining = true;
            static readonly Random rng = new Random();
            public DropoutLayer(double dropoutRate) { rate = dropoutRate; }

            public void SetTrainingMode(bool training) => isTraining = training;

            public override double[] Forward(double[] input)
            {
                if (!isTraining)
                {
                    return input;
                }

                mask = new bool[input.Length];
                double[] output = new double[input.Length];
                double scaleFactor = 1.0 / (1.0 - rate);
                for (int i = 0; i < input.Length; i++)
                {
                    mask[i] = rng.NextDouble() >= rate;
                    output[i] = mask[i] ? input[i] * scaleFactor : 0.0;
                }
                return output;
            }
            public override double[] Backpropagate(double[] error, double lr)
            {
                if (!isTraining || mask == null) return error;

                double[] back = new double[error.Length];
                double scaleFactor = 1.0 / (1.0 - rate);
                for (int i = 0; i < error.Length; i++) back[i] = mask[i] ? error[i] * scaleFactor : 0.0;
                return back;
            }
            public override object Save() => new { type = "DropoutLayer", rate };
            public override void Load(JsonElement data) { }
        }


        class NeuralNetwork
        {
            readonly List<Layer> layers;

            public NeuralNetwork()
            {
                layers = new List<Layer>
                {
                    new DenseLayer(TOTAL_INPUT_FEATURES, 128, "relu"),
                    new DropoutLayer(0.25),
                    new DenseLayer(128, 64, "relu"),
                    new DropoutLayer(0.25),
                    new DenseLayer(64, 3, "tanh")
                };
            }

            private void SetDropoutTrainingMode(bool isTraining)
            {
                foreach (var layer in layers)
                {
                    if (layer is DropoutLayer dropoutLayer)
                    {
                        dropoutLayer.SetTrainingMode(isTraining);
                    }
                }
            }

            public double[] Forward(double[] x, bool isTrainingContext = false)
            {
                SetDropoutTrainingMode(isTrainingContext);
                double[] cur = x;
                foreach (var l in layers) cur = l.Forward(cur);
                return cur;
            }

            public void Train(double[][] X, double[][] Y, int epochs, double lr, Action<int, double> cb = null)
            {
                for (int ep = 0; ep < epochs; ep++)
                {
                    double loss = 0;

                    for (int n = 0; n < X.Length; n++)
                    {
                        var outp = Forward(X[n], true);
                        var err = new double[outp.Length];
                        for (int i = 0; i < err.Length; i++)
                        {
                            err[i] = outp[i] - Y[n][i];
                            loss += err[i] * err[i];
                        }
                        var back = err;
                        for (int i = layers.Count - 1; i >= 0; i--)
                            back = layers[i].Backpropagate(back, lr);
                    }
                    cb?.Invoke(ep, loss / X.Length);
                }
                SetDropoutTrainingMode(false);
            }
            public void Save(string path)
            {
                var json = new { layers = layers.Select(l => l.Save()).ToArray() };
                File.WriteAllText(path, JsonSerializer.Serialize(json, new JsonSerializerOptions { WriteIndented = true }));
            }
            public void Load(string path)
            {
                if (!File.Exists(path))
                {
                    Console.WriteLine($"Model dosyası bulunamadı: {path}. Varsayılan ağ yapısı kullanılacak.");
                    return;
                }
                try
                {
                    var doc = JsonDocument.Parse(File.ReadAllText(path));
                    var arr = doc.RootElement.GetProperty("layers").EnumerateArray();
                    layers.Clear();
                    foreach (var elem in arr)
                    {
                        var t = elem.GetProperty("type").GetString();
                        Layer l;
                        if (t == "DenseLayer")
                        {
                            int inSize = elem.GetProperty("inSize").GetInt32();
                            int outSize = elem.GetProperty("outSize").GetInt32();
                            string activation = elem.GetProperty("activation").GetString();

                            if (layers.Count == 0 && inSize != TOTAL_INPUT_FEATURES)
                            {
                                Console.WriteLine($"UYARI: Yüklenen modelin ('{path}') ilk katman giriş boyutu ({inSize}) mevcut kod konfigürasyonuyla ({TOTAL_INPUT_FEATURES}) eşleşmiyor! Bu kritik bir hataya yol açabilir.");
                                throw new InvalidOperationException("Yüklenen modelin giriş boyutu uyumsuz.");
                            }
                            l = new DenseLayer(inSize, outSize, activation);
                        }
                        else if (t == "DropoutLayer")
                        {
                            l = new DropoutLayer(elem.GetProperty("rate").GetDouble());
                        }
                        else
                        {
                            throw new NotSupportedException($"Bilinmeyen katman tipi: {t}");
                        }
                        l.Load(elem);
                        layers.Add(l);
                    }
                    Console.WriteLine($"Model başarıyla yüklendi: {path}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Model yüklenirken hata oluştu ({path}): {ex.Message}. Varsayılan ağ yapısı kullanılacak.");
                    layers.Clear();
                    layers.Add(new DenseLayer(TOTAL_INPUT_FEATURES, 128, "relu"));
                    layers.Add(new DropoutLayer(0.25));
                    layers.Add(new DenseLayer(128, 64, "relu"));
                    layers.Add(new DropoutLayer(0.25));
                    layers.Add(new DenseLayer(64, 3, "tanh"));
                }
                SetDropoutTrainingMode(false);
            }
        }

        readonly NeuralNetwork net = new NeuralNetwork();
        readonly List<(double[] feats, double[] tgt)> trainingData = new List<(double[] feats, double[] tgt)>();
        readonly Queue<PointF> mouseTrail = new Queue<PointF>(MAX_POINTS);

        PointF targetPos;
        bool autoMove = false;
        readonly Random rng = new Random();
        readonly Stopwatch fpsWatch = Stopwatch.StartNew();
        int frameCounter = 0;
        bool trainingInProgress = false;

        PointF? _last_ai_pos = null;
        Queue<PointF> _ai_pos_history_for_path_drawing = new Queue<PointF>(MAX_POINTS);


        public PredictorForm()
        {
            ClientSize = new Size(WIDTH, HEIGHT);
            Text = "Fare Tahmincisi (Gelişmiş Sıralı MLP) – .NET 4.8";
            DoubleBuffered = true;
            KeyPreview = true;
            targetPos = RandomTarget();

            PointF initialMousePos = PointToClient(Cursor.Position);
            for (int i = 0; i < HISTORY_POINTS_NEEDED_FOR_FEATURES; ++i) mouseTrail.Enqueue(initialMousePos);


            var timer = new Timer { Interval = Math.Max(1, 1000 / FPS) };
            timer.Tick += (_, __) => { UpdateLogic(); Invalidate(); };
            timer.Start();

            Paint += OnPaintCanvas;
            KeyDown += OnKeyDown;
            Load += PredictorForm_Load;
        }
        private void PredictorForm_Load(object sender, EventArgs e)
        {
            if (!TRAINING_MODE && File.Exists("mouse_predictor_vseq_adv.json"))
            {
                Console.WriteLine("Test modu, mouse_predictor_vseq_adv.json yükleniyor...");
                net.Load("mouse_predictor_vseq_adv.json");
            }
            else if (TRAINING_MODE)
            {
                Console.WriteLine("Eğitim modu, yeni model oluşturulacak veya üzerine yazılacak.");
            }
            else
            {
                Console.WriteLine("Test modu, ancak mouse_predictor_vseq_adv.json bulunamadı. Varsayılan ağ kullanılacak.");
            }

            ShowMessage(TRAINING_MODE
                ? "Eğitim modundayız (Gelişmiş Sıralı MLP). Fareyi hedeflere sürükle! (S: Eğit ve Kaydet)"
                : "Test modundayız (Gelişmiş Sıralı MLP). V ile otomatik hareketi aç/kapat. (P: Tahmini Yolu Detaylı Çiz)");
        }


        PointF RandomTarget() => new PointF(rng.Next(TARGET_RADIUS, WIDTH - TARGET_RADIUS), rng.Next(TARGET_RADIUS, HEIGHT - TARGET_RADIUS));
        static PointF Clamp(PointF p, int w, int h) => new PointF(Math.Max(0, Math.Min(w, p.X)), Math.Max(0, Math.Min(h, p.Y)));
        void ShowMessage(string msg) => MessageBox.Show(this, msg, Text, MessageBoxButtons.OK, MessageBoxIcon.Information);
        double Distance(PointF a, PointF b) { double dx = a.X - b.X, dy = a.Y - b.Y; return Math.Sqrt(dx * dx + dy * dy); }
        static double[] Vec(params double[] v) => v;

        double[] GetFeaturesForOneTimeStep(PointF pk, PointF pk1, PointF pk2, PointF pk3, PointF currentTargetPos)
        {
            double[] features = new double[FEATURES_PER_TIMESTEP];

            features[0] = (currentTargetPos.X - pk.X) / WIDTH;
            features[1] = (currentTargetPos.Y - pk.Y) / HEIGHT;

            double[] move_vec_k = { pk.X - pk1.X, pk.Y - pk1.Y };
            double speed_k = Math.Sqrt(move_vec_k[0] * move_vec_k[0] + move_vec_k[1] * move_vec_k[1]);
            features[2] = (speed_k > 0.001) ? move_vec_k[0] / speed_k : 0;
            features[3] = (speed_k > 0.001) ? move_vec_k[1] / speed_k : 0;
            features[4] = Math.Min(1.0, speed_k / MAX_SPEED_INPUT_NORMALIZATION);

            double[] move_vec_k1 = { pk1.X - pk2.X, pk1.Y - pk2.Y };
            double speed_k1 = Math.Sqrt(move_vec_k1[0] * move_vec_k1[0] + move_vec_k1[1] * move_vec_k1[1]);
            features[5] = Math.Min(1.0, speed_k1 / MAX_SPEED_INPUT_NORMALIZATION);

            double acceleration_raw = speed_k - speed_k1;
            features[6] = Math.Max(-1.0, Math.Min(1.0, acceleration_raw / MAX_ACCELERATION_NORMALIZATION));

            double angular_change_k_k1 = 0;
            if (speed_k > 0.001 && speed_k1 > 0.001)
            {
                double dot_product = (move_vec_k[0] * move_vec_k1[0] + move_vec_k[1] * move_vec_k1[1]) / (speed_k * speed_k1);
                dot_product = Math.Max(-1.0, Math.Min(1.0, dot_product));
                double angle_rad = Math.Acos(dot_product);
                double cross_product_z = move_vec_k[0] * move_vec_k1[1] - move_vec_k[1] * move_vec_k1[0];
                if (cross_product_z < 0) angle_rad = -angle_rad;
                angular_change_k_k1 = angle_rad;
            }
            features[7] = Math.Max(-1.0, Math.Min(1.0, angular_change_k_k1 / ANGULAR_CHANGE_NORMALIZATION_DIVISOR));

            double[] move_vec_k2 = { pk2.X - pk3.X, pk2.Y - pk3.Y };
            double speed_k2 = Math.Sqrt(move_vec_k2[0] * move_vec_k2[0] + move_vec_k2[1] * move_vec_k2[1]);
            double angular_change_k1_k2 = 0;
            if (speed_k1 > 0.001 && speed_k2 > 0.001)
            {
                double dot_product = (move_vec_k1[0] * move_vec_k2[0] + move_vec_k1[1] * move_vec_k2[1]) / (speed_k1 * speed_k2);
                dot_product = Math.Max(-1.0, Math.Min(1.0, dot_product));
                double angle_rad = Math.Acos(dot_product);
                double cross_product_z = move_vec_k1[0] * move_vec_k2[1] - move_vec_k1[1] * move_vec_k2[0];
                if (cross_product_z < 0) angle_rad = -angle_rad;
                angular_change_k1_k2 = angle_rad;
            }
            features[8] = Math.Max(-1.0, Math.Min(1.0, angular_change_k1_k2 / ANGULAR_CHANGE_NORMALIZATION_DIVISOR));

            return features;
        }

        PointF GetPointFromTrailSafely(IReadOnlyList<PointF> trail, int index, PointF defaultIfEmpty)
        {
            if (trail == null || trail.Count == 0) return defaultIfEmpty;
            if (index < 0) return trail[0];
            if (index >= trail.Count) return trail[trail.Count - 1];
            return trail[index];
        }


        double[] PrepareFeatureVector(IReadOnlyList<PointF> fullTrail, PointF currentTargetPos)
        {
            List<double> all_input_features = new List<double>(TOTAL_INPUT_FEATURES);
            PointF default_padding_point = (fullTrail != null && fullTrail.Count > 0) ? fullTrail[0] : PointF.Empty;

            for (int i = 0; i < MOUSE_HISTORY_SEQUENCE_LENGTH; i++)
            {
                PointF pk = GetPointFromTrailSafely(fullTrail, fullTrail.Count - 1 - i, default_padding_point);
                PointF pk1 = GetPointFromTrailSafely(fullTrail, fullTrail.Count - 1 - i - 1, default_padding_point);
                PointF pk2 = GetPointFromTrailSafely(fullTrail, fullTrail.Count - 1 - i - 2, default_padding_point);
                PointF pk3 = GetPointFromTrailSafely(fullTrail, fullTrail.Count - 1 - i - 3, default_padding_point);

                all_input_features.AddRange(GetFeaturesForOneTimeStep(pk, pk1, pk2, pk3, currentTargetPos));
            }

            if (all_input_features.Count != TOTAL_INPUT_FEATURES)
            {
                Console.WriteLine($"UYARI: Özellik vektörü boyutu ({all_input_features.Count}) beklenenden ({TOTAL_INPUT_FEATURES}) farklı! Bu bir hata olmalı. Padding eksik olabilir.");
                while (all_input_features.Count < TOTAL_INPUT_FEATURES && all_input_features.Count > 0) all_input_features.Add(0.0);
                if (all_input_features.Count > TOTAL_INPUT_FEATURES) all_input_features.RemoveRange(TOTAL_INPUT_FEATURES, all_input_features.Count - TOTAL_INPUT_FEATURES);
            }
            return all_input_features.ToArray();
        }


        void UpdateLogic()
        {
            var os_mouse_pos = PointToClient(Cursor.Position);

            mouseTrail.Enqueue(os_mouse_pos);
            while (mouseTrail.Count > MAX_POINTS) mouseTrail.Dequeue();

            if (autoMove && _last_ai_pos.HasValue)
            {
                _ai_pos_history_for_path_drawing.Enqueue(_last_ai_pos.Value);
                while (_ai_pos_history_for_path_drawing.Count > MAX_POINTS) _ai_pos_history_for_path_drawing.Dequeue();
            }


            PointF current_pos_for_target_check = autoMove && _last_ai_pos.HasValue ? _last_ai_pos.Value : os_mouse_pos;
            if (Distance(current_pos_for_target_check, targetPos) < TARGET_RADIUS)
            {
                targetPos = RandomTarget();
            }

            if (TRAINING_MODE)
            {
                if (mouseTrail.Count >= 2)
                {
                    PointF current_for_tgt = mouseTrail.ElementAt(mouseTrail.Count - 1);
                    PointF prev_for_tgt = mouseTrail.ElementAt(mouseTrail.Count - 2);
                    CollectSample(current_for_tgt, prev_for_tgt, mouseTrail.ToList(), targetPos);
                }
            }
            else if (autoMove)
            {
                if (mouseTrail.Count >= HISTORY_POINTS_NEEDED_FOR_FEATURES)
                {
                    List<PointF> trail_for_prediction;
                    if (_ai_pos_history_for_path_drawing.Count >= HISTORY_POINTS_NEEDED_FOR_FEATURES)
                    {
                        trail_for_prediction = _ai_pos_history_for_path_drawing.ToList();
                    }
                    else
                    {
                        trail_for_prediction = mouseTrail.ToList();
                    }


                    var prediction = PredictNextPoint(trail_for_prediction, targetPos);
                    var next_predicted_pos = prediction.NextPosition;

                    Cursor.Position = PointToScreen(Point.Round(next_predicted_pos));
                    _last_ai_pos = next_predicted_pos;
                }
            }
            else
            {
                _last_ai_pos = null;
                _ai_pos_history_for_path_drawing.Clear();
                _ai_pos_history_for_path_drawing.Enqueue(os_mouse_pos);
                while (_ai_pos_history_for_path_drawing.Count > MAX_POINTS) _ai_pos_history_for_path_drawing.Dequeue();

            }

            frameCounter++;
            if (fpsWatch.ElapsedMilliseconds >= 1000)
            {
                Text = $"Fare Tahmincisi (Gelişmiş) – FPS: {frameCounter}";
                frameCounter = 0;
                fpsWatch.Restart();
            }
        }

        void CollectSample(PointF current_mouse_pos, PointF prev_mouse_pos, IReadOnlyList<PointF> trail_for_features, PointF currentTargetPos)
        {
            if (trail_for_features.Count < HISTORY_POINTS_NEEDED_FOR_FEATURES) return;

            var feats = PrepareFeatureVector(trail_for_features, currentTargetPos);

            double[] dist_to_target_vec = { currentTargetPos.X - current_mouse_pos.X, currentTargetPos.Y - current_mouse_pos.Y };
            double target_dir_mag = Math.Sqrt(dist_to_target_vec[0] * dist_to_target_vec[0] + dist_to_target_vec[1] * dist_to_target_vec[1]);
            double target_ideal_dir_x = (target_dir_mag > 0.001) ? dist_to_target_vec[0] / target_dir_mag : 0;
            double target_ideal_dir_y = (target_dir_mag > 0.001) ? dist_to_target_vec[1] / target_dir_mag : 0;

            double[] last_move_vec = { current_mouse_pos.X - prev_mouse_pos.X, current_mouse_pos.Y - prev_mouse_pos.Y };
            double last_move_mag = Math.Sqrt(last_move_vec[0] * last_move_vec[0] + last_move_vec[1] * last_move_vec[1]);

            double speed_ratio_to_base = (MOUSE_STEP_SIZE > 0.001) ? last_move_mag / MOUSE_STEP_SIZE : 0;
            double tanh_target_speed = (2.0 * Math.Min(Math.Max(0, speed_ratio_to_base), MAX_SPEED_MULTIPLIER) / MAX_SPEED_MULTIPLIER) - 1.0;
            tanh_target_speed = Math.Max(-1.0, Math.Min(1.0, tanh_target_speed));

            var tgt = Vec(target_ideal_dir_x, target_ideal_dir_y, tanh_target_speed);

            if (last_move_mag > 0.1)
            {
                trainingData.Add((feats, tgt));
            }
        }

        (PointF NextPosition, double SpeedMagnitude) PredictNextPoint(IReadOnlyList<PointF> trail, PointF currentTargetPos)
        {
            if (trail.Count < HISTORY_POINTS_NEEDED_FOR_FEATURES)
            {
                return (trail.LastOrDefault(), 0);
            }

            var feats_for_pred = PrepareFeatureVector(trail, currentTargetPos);
            var pred_output = net.Forward(feats_for_pred, false);

            double predicted_dir_x_raw = pred_output[0];
            double predicted_dir_y_raw = pred_output[1];
            double predicted_tanh_speed = pred_output[2];

            double pred_dir_mag = Math.Sqrt(predicted_dir_x_raw * predicted_dir_x_raw + predicted_dir_y_raw * predicted_dir_y_raw);
            double final_predicted_dir_x = 0;
            double final_predicted_dir_y = 0;
            if (pred_dir_mag > 0.001)
            {
                final_predicted_dir_x = predicted_dir_x_raw / pred_dir_mag;
                final_predicted_dir_y = predicted_dir_y_raw / pred_dir_mag;
            }

            double speed_multiplier_normalized_01 = (predicted_tanh_speed + 1.0) / 2.0;
            double actual_speed_multiplier = MIN_SPEED_MULTIPLIER + speed_multiplier_normalized_01 * (MAX_SPEED_MULTIPLIER - MIN_SPEED_MULTIPLIER);
            double actual_step_magnitude = actual_speed_multiplier * MOUSE_STEP_SIZE;

            PointF current_pos_from_trail = trail.Last();
            var next_pos_calculated = new PointF(
                (float)(current_pos_from_trail.X + final_predicted_dir_x * actual_step_magnitude),
                (float)(current_pos_from_trail.Y + final_predicted_dir_y * actual_step_magnitude));

            return (Clamp(next_pos_calculated, WIDTH, HEIGHT), actual_step_magnitude);
        }

        List<(PointF Point, double SpeedForNextSegment)> GetPredictedPath(
             IReadOnlyList<PointF> initial_trail_segment,
             PointF pathTargetPos,
             int maxSteps = 100)
        {
            var path_data = new List<(PointF Point, double SpeedForNextSegment)>();
            if (initial_trail_segment == null || initial_trail_segment.Count == 0) return path_data;

            Queue<PointF> simulated_trail_queue = new Queue<PointF>(initial_trail_segment);
            PointF current_sim_pos = initial_trail_segment.Last();

            if (maxSteps <= 0)
            {
                path_data.Add((current_sim_pos, 0.0));
                return path_data;
            }

            for (int i = 0; i < maxSteps; i++)
            {
                if (simulated_trail_queue.Count < HISTORY_POINTS_NEEDED_FOR_FEATURES)
                {
                }


                var predictionResult = PredictNextPoint(simulated_trail_queue.ToList(), pathTargetPos);
                PointF next_pos = predictionResult.NextPosition;
                double speed_of_segment = predictionResult.SpeedMagnitude;

                path_data.Add((current_sim_pos, speed_of_segment));

                simulated_trail_queue.Enqueue(next_pos);
                while (simulated_trail_queue.Count > MAX_POINTS)
                {
                    simulated_trail_queue.Dequeue();
                }

                bool stuck = Distance(current_sim_pos, next_pos) < 0.1 && i > 10;
                if (Distance(next_pos, pathTargetPos) < TARGET_RADIUS || i == maxSteps - 1 || stuck)
                {
                    if (stuck && path_data.Count > 0)
                    {
                        path_data[path_data.Count - 1] = (path_data.Last().Point, 0.0);
                    }
                    else
                    {
                        path_data.Add((next_pos, 0.0));
                    }
                    break;
                }
                current_sim_pos = next_pos;
            }
            return path_data;
        }

        void OnPaintCanvas(object sender, PaintEventArgs e)
        {
            var g = e.Graphics;
            g.Clear(Color.FromArgb(20, 20, 20));

            using (var redBrush = new SolidBrush(Color.Tomato))
                g.FillEllipse(redBrush, targetPos.X - TARGET_RADIUS, targetPos.Y - TARGET_RADIUS, TARGET_RADIUS * 2, TARGET_RADIUS * 2);

            PointF actual_cursor_pos = PointToClient(Cursor.Position);
            using (var blueBrush = new SolidBrush(Color.SkyBlue))
                g.FillEllipse(blueBrush, actual_cursor_pos.X - 7, actual_cursor_pos.Y - 7, 14, 14);

            if (autoMove && _last_ai_pos.HasValue)
            {
                using (var aiCursorBrush = new SolidBrush(Color.LimeGreen))
                    g.FillEllipse(aiCursorBrush, _last_ai_pos.Value.X - 5, _last_ai_pos.Value.Y - 5, 10, 10);

                if (_ai_pos_history_for_path_drawing.Count >= HISTORY_POINTS_NEEDED_FOR_FEATURES)
                {
                    var predicted_path_with_speeds = GetPredictedPath(_ai_pos_history_for_path_drawing.ToList(), targetPos, 50);
                    if (predicted_path_with_speeds.Count > 1)
                    {
                        var predicted_path_points = predicted_path_with_speeds.Select(data => data.Point).ToArray();
                        using (var pathPen = new Pen(Color.FromArgb(150, Color.LightGreen), 2))
                            g.DrawLines(pathPen, predicted_path_points);
                    }
                }
            }

            if (TRAINING_MODE && mouseTrail.Count > 1)
            {
                using (var trailPen = new Pen(Color.FromArgb(100, 0, 200, 0), 2))
                {
                    PointF[] pts = mouseTrail.ToArray();
                    int startIdx = Math.Max(0, pts.Length - 50);
                    for (int i = startIdx + 1; i < pts.Length; i++)
                    {
                        g.DrawLine(trailPen, pts[i - 1], pts[i]);
                    }
                }
            }
        }
        void OnKeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.V && !TRAINING_MODE)
            {
                autoMove = !autoMove;
                if (autoMove)
                {
                    _last_ai_pos = PointToClient(Cursor.Position);
                    _ai_pos_history_for_path_drawing.Clear();
                    PointF currentMouse = PointToClient(Cursor.Position);
                    for (int i = 0; i < HISTORY_POINTS_NEEDED_FOR_FEATURES; ++i) _ai_pos_history_for_path_drawing.Enqueue(currentMouse);

                }
                else
                {
                    _last_ai_pos = null;
                }
                Invalidate();
            }
            if (e.KeyCode == Keys.S && TRAINING_MODE && trainingData.Count > 0 && !trainingInProgress)
            {
                trainingInProgress = true;
                Task.Run(() => RunTraining());
            }
            if (e.KeyCode == Keys.P && !TRAINING_MODE && !autoMove)
            {
                List<PointF> initial_path_trail = new List<PointF>();
                PointF currentMouseForP = PointToClient(Cursor.Position);

                if (_ai_pos_history_for_path_drawing.Count >= HISTORY_POINTS_NEEDED_FOR_FEATURES)
                {
                    initial_path_trail = _ai_pos_history_for_path_drawing.ToList();
                }
                else
                {
                    initial_path_trail = mouseTrail.Take(HISTORY_POINTS_NEEDED_FOR_FEATURES - 1).ToList();
                    initial_path_trail.Add(currentMouseForP);
                    while (initial_path_trail.Count < HISTORY_POINTS_NEEDED_FOR_FEATURES) initial_path_trail.Insert(0, initial_path_trail[0]);
                }


                var sw = new Stopwatch();
                sw.Start();
                var path_data_with_speeds = GetPredictedPath(initial_path_trail, targetPos, 100);
                sw.Stop();
                Console.WriteLine($"GetPredictedPath süresi: {sw.Elapsed.TotalMilliseconds:F2} ms, {path_data_with_speeds.Count} nokta.");
                Console.WriteLine($"=== Tahmini Yol (P tuşu) ({path_data_with_speeds.Count} nokta/segment) ===");
                for (int i = 0; i < path_data_with_speeds.Count; i++)
                {
                    var data = path_data_with_speeds[i];
                    Console.WriteLine($"{i,3}: Nokta ({data.Point.X:F1}, {data.Point.Y:F1}) -> Sonraki Segment Hızı: {data.SpeedForNextSegment:F2} piksel/kare");
                }
            }
        }

        private void InitializeComponent()
        {
            this.SuspendLayout();
            // 
            // PredictorForm
            // 
            this.ClientSize = new System.Drawing.Size(284, 261);
            this.Name = "PredictorForm";
            this.Load += new System.EventHandler(this.PredictorForm_Load_1);
            this.ResumeLayout(false);

        }

        private void PredictorForm_Load_1(object sender, EventArgs e)
        {

        }

        void RunTraining()
        {
            if (trainingData.Count == 0)
            {
                BeginInvoke((Action)(() => ShowMessage("Eğitilecek veri yok!")));
                trainingInProgress = false;
                return;
            }
            var X = trainingData.Select(d => d.feats).ToArray();
            var Y = trainingData.Select(d => d.tgt).ToArray();

            Invoke((Action)(() => ShowMessage($"Eğitim başlıyor… Epoch sayısı: {X.Length}")));
            net.Train(X, Y, 800, 0.0005,
                (ep, loss) => {
                    if (ep % 10 == 0)
                        BeginInvoke((Action)(() =>
                            Text = $"Eğitim {ep}/800 – Loss {loss:F6}"));
                });
            net.Save("mouse_predictor_vseq_adv.json");
            BeginInvoke((Action)(() =>
            {
                ShowMessage("Eğitim bitti ve model 'mouse_predictor_vseq_adv.json' olarak kaydedildi!");
                trainingInProgress = false;
                trainingData.Clear();
            }));
        }
    }
}
