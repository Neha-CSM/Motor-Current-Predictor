import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import onnxmltools
import time
import onnxruntime as ort
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score
from xgboost import XGBRegressor
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import IsolationForest

IMU_MOTION_INPUTS = [
    'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',  
    'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z'                    
]
MOTOR_TARGETS = ['m1current', 'm2current', 'm3current', 'm4current']

def run_motor_analysis(df, target_name):
    print(f"\nStarting Analysis for: {target_name}")

    # data cleaning and type conversion
    work_df = df.copy()
    cols_to_clean = IMU_MOTION_INPUTS + [target_name]
    work_df[cols_to_clean] = work_df[cols_to_clean].apply(pd.to_numeric, errors='coerce')
    work_df = work_df.dropna(subset=cols_to_clean).astype(np.float32)

    # anomaly detection using isolation forest
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    iso_preds = iso_forest.fit_predict(work_df[IMU_MOTION_INPUTS])
    
    # separate the anomalies for logging purposes
    anomalies_only = work_df[iso_preds == -1]
    anomaly_count = len(anomalies_only)
    print(f"Detected {anomaly_count} anomalies in the raw data.")
    anomalies_only.to_csv(f"{target_name}_detected_anomalies.csv", index=False)
    
    # keep normal observations
    clean_df = work_df[iso_preds == 1].copy()
    print(f"Removed {anomaly_count} anomalies. New dataset size: {len(clean_df)} rows")

    # training and testing split
    X_raw = clean_df[IMU_MOTION_INPUTS].values
    y_raw = clean_df[target_name].values
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    # xgboost model configuration and training
    model = XGBRegressor(
        tree_method='hist',       
        n_jobs=-1,                
        n_estimators=150,         
        max_depth=6,             
        learning_rate=0.2,       
        min_child_weight=5,       
        
        objective='reg:squarederror',
        random_state=42
    )
    # train the model
    model.fit(X_train, y_train)

    #model evaluation
    y_pred = model.predict(X_test)

    # calculate root mean squared error and r-squared score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Performance for {target_name}:")
    print(f"  RMSE: {rmse:.4f} Amps")
    print(f"  R-squared: {r2:.4f}")

    # convert the tree-based xgboost model into a standardized math graph
    export_to_onnx(model, target_name, len(IMU_MOTION_INPUTS))
    
    # print the importance of each sensor axis for this specific motor
    importances = pd.Series(model.feature_importances_, index=IMU_MOTION_INPUTS)
    print("Feature Importance Ranking:")
    print(importances.sort_values(ascending=False).to_string())

    # onnx session configuration and diagnosis
    print(f"ONNX Conversion Diagnosis for {target_name}")
    
    # initialize session options for graph optimization
    sess_options = ort.SessionOptions()
    # enable full graph optimizations to fuse math operations
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL 
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # create the inference session from the saved .onnx file
    onnx_file = f"{target_name}_model.onnx"
    session = ort.InferenceSession(onnx_file, sess_options)
    # identify the input node name for the model
    input_name = session.get_inputs()[0].name
    
    # run predictions through both engines to verify math consistency
    xgb_preds = model.predict(X_test).flatten()
    onnx_preds = session.run(None, {input_name: X_test})[0].flatten()
    
    # check if the results are within 0.1% of each other
    tolerance = 0.001
    relative_error = np.abs(xgb_preds - onnx_preds) / (np.abs(xgb_preds) + 1e-6)
    is_correct = relative_error < tolerance
    
    accuracy_pct = np.mean(is_correct) * 100
    print(f"  Conversion Accuracy: {accuracy_pct:.2f}%")
    
    if accuracy_pct > 99.9:
        print("  Status: PASSED. ONNX is a near-perfect clone.")
    else:
        print(f"  Status: WARNING. Math drift detected.")

    # generate and save a scatter plot of predicted vs actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"Actual vs Predicted Current: {target_name}")
    plt.xlabel("Actual Current (Amps)")
    plt.ylabel("Predicted Current (Amps)")
    plt.savefig(f"{target_name}_performance.png")
    plt.close()

    # step 9: speed benchmarking
    print(f"ONNX vs XGBoost Speed/Precision Test")
    # compare xgboost and onnx
    xgb_preds_bench, onnx_preds_bench = benchmark_models(model, session, X_test, input_name)
    
    # verify the absolute difference in ampere predictions
    avg_diff = np.mean(np.abs(xgb_preds_bench.flatten() - onnx_preds_bench))
    print(f"  Avg Accuracy Difference: {avg_diff:.10f} Amps\n")

def export_to_onnx(model, target_name, num_features):
    # define the input tensor shape (any number of rows, 6 columns)
    initial_type = [('float_input', FloatTensorType([None, num_features]))]
    try:
        # translate the tree logic into the onnx format
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type, target_opset=12)
        # save the binary model file
        onnxmltools.utils.save_model(onnx_model, f"{target_name}_model.onnx")
        print(f"Successfully exported {target_name} to {target_name}_model.onnx")
    except Exception as e:
        print(f"Failed export: {e}")

def benchmark_models(model, session, X_test, input_name, n_runs=20, batch_multiplier=10):
    # repeat the test data to create a larger workload for stable timing
    X_large = np.tile(X_test, (batch_multiplier, 1))
    print(f"  Benchmarking with batch size: {len(X_large)} samples")
    
    # warmup iterations to fill caches and stabilize cpu clock speed
    for _ in range(5):
        model.predict(X_large)
        session.run(None, {input_name: X_large})
    
    xgb_times, onnx_times = [], []
    for _ in range(n_runs):
        # time native xgboost prediction
        start = time.perf_counter()
        xgb_res = model.predict(X_large)
        xgb_times.append(time.perf_counter() - start)
        
        # time onnx runtime prediction
        start = time.perf_counter()
        onnx_res = session.run(None, {input_name: X_large})[0]
        onnx_times.append(time.perf_counter() - start)
    
    # calculate median and standard deviation with times 
    xgb_median = np.median(xgb_times)
    onnx_median = np.median(onnx_times)
    xgb_std = np.std(xgb_times)
    onnx_std = np.std(onnx_times)
    
    # display results with +/- standard deviation
    print(f"  XGBoost: {xgb_median*1000:.2f}ms (std: {xgb_std*1000:.2f}ms)")
    print(f"  ONNX:    {onnx_median*1000:.2f}ms (std: {onnx_std*1000:.2f}ms)")
    print(f"  Speedup: {xgb_median/onnx_median:.2f}x")
    
    # return predictions from the first batch only for accuracy comparison
    return xgb_res[:len(X_test)], onnx_res[:len(X_test)].flatten()

if __name__ == "__main__":
    try:
        # load the telemetry data
        data = pd.read_csv("clean_output.csv")
        # iterate through each motor to train independent digital twins
        for motor in MOTOR_TARGETS:
            if motor in data.columns:
                run_motor_analysis(data, motor)
        print("All analyses complete.")
    except FileNotFoundError:
        print("File clean_output.csv not found.")