import streamlit as st
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import psutil
import os
import gc

st.set_page_config(layout="wide")
st.title("🚀 Streamlit Cloud Resource Limitations Test")
st.markdown("This app demonstrates resource usage with multiple ML pipelines")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Pipeline configurations
    st.subheader("Pipeline Settings")
    num_samples = st.slider("Training Samples", 1000, 50000, 10000, step=1000)
    num_features = st.slider("Number of Features", 10, 500, 100, step=10)
    
    # Model complexity
    st.subheader("Model Complexity")
    rf_depth = st.slider("Random Forest Depth", 5, 50, 20)
    gb_trees = st.slider("Gradient Boosting Trees", 50, 500, 100)
    svm_complexity = st.slider("SVM Complexity (C)", 0.1, 10.0, 1.0)
    nn_layers = st.slider("Neural Network Layers", 1, 5, 3)
    
    # Memory allocation test
    st.subheader("Memory Test")
    memory_mb = st.slider("Allocate Memory (MB)", 10, 500, 100)
    
    # Run options
    run_all = st.checkbox("Run All Pipelines", value=True)
    stress_test = st.checkbox("Enable Stress Test", value=False)
    
    if st.button("🚨 Run Resource Test"):
        test_running = True
    else:
        test_running = False

# Function to display system resources
def display_resources():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Memory Usage", f"{psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")
    
    with col2:
        st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
    
    with col3:
        # Create some temporary data to show memory pressure
        if stress_test:
            large_array = np.random.rand(1000, 1000)
            st.metric("Temp Array Size", f"{large_array.nbytes / 1024 / 1024:.1f} MB")

# Function to create memory pressure
def allocate_memory(size_mb):
    """Allocate memory to test limits"""
    bytes_to_allocate = size_mb * 1024 * 1024
    try:
        # Create a large array
        array_size = bytes_to_allocate // 8  # 8 bytes per float64
        large_array = np.random.rand(array_size)
        return large_array.nbytes / 1024 / 1024
    except MemoryError:
        return "Memory Error!"

# Pipeline 1: Random Forest with varying complexity
def pipeline_random_forest(X, y, max_depth):
    st.subheader("🌲 Pipeline 1: Random Forest")
    
    start_time = time.time()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model with varying complexity
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    with st.spinner(f"Training Random Forest with depth {max_depth}..."):
        model.fit(X_train, y_train)
    
    # Simulate prediction on large dataset
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    
    end_time = time.time()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Training Time", f"{end_time - start_time:.2f}s")
    
    return accuracy

# Pipeline 2: Gradient Boosting
def pipeline_gradient_boosting(X, y, n_estimators):
    st.subheader("📈 Pipeline 2: Gradient Boosting")
    
    start_time = time.time()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=0.1,
        random_state=42
    )
    
    with st.spinner(f"Training Gradient Boosting with {n_estimators} trees..."):
        model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    
    end_time = time.time()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Training Time", f"{end_time - start_time:.2f}s")
    
    return accuracy

# Pipeline 3: Support Vector Machine
def pipeline_svm(X, y, C_value):
    st.subheader("🔍 Pipeline 3: Support Vector Machine")
    
    start_time = time.time()
    
    # Use smaller subset for SVM (can be memory intensive)
    sample_size = min(5000, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample, y_sample = X[indices], y[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42
    )
    
    model = SVC(C=C_value, kernel='rbf', random_state=42)
    
    with st.spinner(f"Training SVM with C={C_value}..."):
        model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    
    end_time = time.time()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Training Time", f"{end_time - start_time:.2f}s")
    
    return accuracy

# Pipeline 4: Neural Network
def pipeline_neural_network(X, y, hidden_layers):
    st.subheader("🧠 Pipeline 4: Neural Network")
    
    start_time = time.time()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create hidden layer structure
    hidden_layer_sizes = tuple([100] * hidden_layers)
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=100,
        random_state=42
    )
    
    with st.spinner(f"Training Neural Network with {hidden_layers} hidden layers..."):
        model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    
    end_time = time.time()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Training Time", f"{end_time - start_time:.2f}s")
    
    return accuracy

# Main execution
if test_running:
    st.header("📊 Running Resource Tests")
    
    # Display current resources
    display_resources()
    
    # Create synthetic dataset
    with st.spinner(f"Generating dataset with {num_samples} samples and {num_features} features..."):
        X, y = make_classification(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=num_features // 2,
            n_redundant=num_features // 4,
            random_state=42
        )
    
    st.success(f"✅ Generated dataset: {X.shape[0]} samples × {X.shape[1]} features")
    
    # Memory allocation test
    if stress_test:
        st.subheader("💾 Memory Stress Test")
        with st.spinner(f"Allocating {memory_mb} MB of memory..."):
            allocated = allocate_memory(memory_mb)
            if isinstance(allocated, str):
                st.error(allocated)
            else:
                st.info(f"Allocated {allocated:.1f} MB")
    
    # Run selected pipelines
    results = {}
    
    if run_all or st.checkbox("Run Random Forest"):
        results['Random Forest'] = pipeline_random_forest(X, y, rf_depth)
        display_resources()
        gc.collect()  # Force garbage collection
    
    if run_all or st.checkbox("Run Gradient Boosting"):
        results['Gradient Boosting'] = pipeline_gradient_boosting(X, y, gb_trees)
        display_resources()
        gc.collect()
    
    if run_all or st.checkbox("Run SVM"):
        results['SVM'] = pipeline_svm(X, y, svm_complexity)
        display_resources()
        gc.collect()
    
    if run_all or st.checkbox("Run Neural Network"):
        results['Neural Network'] = pipeline_neural_network(X, y, nn_layers)
        display_resources()
        gc.collect()
    
    # Display results summary
    if results:
        st.header("📈 Results Summary")
        
        # Create summary dataframe
        summary_df = pd.DataFrame({
            'Pipeline': list(results.keys()),
            'Accuracy': list(results.values())
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(summary_df)
        
        with col2:
            fig, ax = plt.subplots()
            ax.bar(results.keys(), results.values())
            ax.set_ylabel('Accuracy')
            ax.set_title('Pipeline Performance Comparison')
            ax.set_ylim([0, 1])
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        # Final resource display
        st.header("📊 Final Resource Usage")
        display_resources()
        
        # Memory info
        st.subheader("Memory Information")
        memory_info = psutil.virtual_memory()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Memory", f"{memory_info.total / 1024 / 1024:.0f} MB")
        with col2:
            st.metric("Used Memory", f"{memory_info.used / 1024 / 1024:.0f} MB")
        with col3:
            st.metric("Memory %", f"{memory_info.percent}%")
    
    st.success("✅ All tests completed!")
    
    # Warning for high memory usage
    if memory_info.percent > 80:
        st.warning("⚠️ High memory usage detected! Streamlit Cloud may restart your app.")
    
    # Tips for Streamlit Cloud
    with st.expander("💡 Streamlit Cloud Tips"):
        st.markdown("""
        **Streamlit Cloud Resource Limits:**
        - Free tier: 1 GB RAM, 1 CPU core
        - Apps restart after 1 hour of inactivity
        - Memory limits are strictly enforced
        
        **Optimization Tips:**
        1. Use smaller datasets
        2. Implement data streaming
        3. Cache expensive computations with `@st.cache_data`
        4. Use simpler models
        5. Clear unused variables with `del` and `gc.collect()`
        6. Avoid loading large models in memory simultaneously
        
        **To test limits:**
        - Increase sample size and features
        - Enable Stress Test
        - Run all pipelines simultaneously
        """)

else:
    st.info("👈 Configure settings in the sidebar and click 'Run Resource Test' to begin")
    
    # Show example configurations
    with st.expander("Example Test Configurations"):
        st.markdown("""
        **Light Test (Free tier friendly):**
        - Training Samples: 5,000
        - Features: 50
        - Memory Allocation: 50 MB
        - Run pipelines one at a time
        
        **Medium Test (May hit limits):**
        - Training Samples: 20,000
        - Features: 200
        - Memory Allocation: 200 MB
        - Run all pipelines
        
        **Stress Test (Likely to hit limits):**
        - Training Samples: 50,000
        - Features: 500
        - Memory Allocation: 500 MB
        - Enable Stress Test checkbox
        - Run all pipelines simultaneously
        """)
    
    # Initial resource display
    st.subheader("Current Resource Status")
    display_resources()
