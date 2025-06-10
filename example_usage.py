"""
Example usage of EHR Embeddings with Google Gemini API
"""
import os
import pandas as pd
import numpy as np
from src.embeddings import GeminiEmbedder, preprocess_ehr_text, create_embedding_features
from src.ml_pipeline import EHRMLPipeline, create_sample_ehr_data
from config import Config

def main():
    """
    Main example demonstrating the EHR embeddings pipeline
    """
    print("EHR Embeddings with Google Gemini API - Example Usage")
    print("=" * 60)
    
    # Check if API key is set
    if not Config.GOOGLE_API_KEY:
        print("⚠️  Warning: GOOGLE_API_KEY not set!")
        print("Please set your Google API key in environment variables or .env file")
        print("For this example, we'll use simulated data instead of real embeddings")
        use_real_api = False
    else:
        use_real_api = True
        print("✅ Google API key found")
    
    # Step 1: Create or load sample EHR data
    print("\n1. Creating sample EHR data...")
    sample_data = create_sample_ehr_data(n_samples=100)
    print(f"Created {len(sample_data)} sample records")
    print("\nSample data preview:")
    print(sample_data[['patient_id', 'chief_complaint', 'diagnosis', 'readmission_risk']].head())
    
    if use_real_api:
        # Step 2: Initialize embedder and generate real embeddings
        print("\n2. Initializing Gemini embedder...")
        embedder = GeminiEmbedder()
        
        # Prepare text data for embedding
        print("3. Preprocessing text data...")
        text_columns = ['chief_complaint', 'diagnosis']
        
        # Combine text fields for embedding
        combined_texts = []
        for _, row in sample_data.iterrows():
            text_parts = [preprocess_ehr_text(str(row[col])) for col in text_columns]
            combined_text = " | ".join(text_parts)
            combined_texts.append(combined_text)
        
        print(f"Example combined text: {combined_texts[0]}")
        
        # Generate embeddings (this will make API calls)
        print("4. Generating embeddings with Gemini API...")
        print("⚠️  This will make API calls and may take some time...")
        
        # For demonstration, only embed first 10 records to save API calls
        small_sample = sample_data.head(10).copy()
        embeddings = embedder.embed_batch(combined_texts[:10])
        
        # Add embeddings to dataframe  
        embedding_list = [embeddings[i] for i in range(len(embeddings))]
        small_sample = small_sample.head(len(embeddings)).copy()
        small_sample['embedding'] = embedding_list
        
        ehr_data = small_sample
        
    else:
        # Use simulated embeddings (already in sample data)
        print("\n2. Using simulated embeddings for demonstration...")
        ehr_data = sample_data
    
    # Step 3: Prepare data for machine learning
    print(f"\n3. Preparing data for machine learning...")
    ml_pipeline = EHRMLPipeline(task_type="classification")
    
    X_train, X_test, y_train, y_test = ml_pipeline.prepare_data(
        df=ehr_data,
        target_column='readmission_risk'
    )
    
    # Step 4: Train multiple models
    print("\n4. Training multiple ML models...")
    training_results = ml_pipeline.train_models(X_train, y_train)
    
    print("\nTraining Results:")
    for model_name, results in training_results.items():
        if "error" not in results:
            acc = results["cv_accuracy_mean"]
            std = results["cv_accuracy_std"]
            print(f"  {model_name:20s}: {acc:.4f} (+/- {std * 2:.4f})")
        else:
            print(f"  {model_name:20s}: Error - {results['error']}")
    
    # Step 5: Evaluate best model
    print(f"\n5. Evaluating best model: {ml_pipeline.best_model_name}")
    test_metrics = ml_pipeline.evaluate_model(ml_pipeline.best_model, X_test, y_test)
    
    print("Test Set Performance:")
    for metric, value in test_metrics.items():
        print(f"  {metric:15s}: {value:.4f}")
    
    # Step 6: Feature importance
    print("\n6. Feature importance analysis...")
    try:
        importance_df = ml_pipeline.get_feature_importance()
        print("Top 10 most important features:")
        print(importance_df.head(10))
    except Exception as e:
        print(f"Could not get feature importance: {e}")
    
    # Step 7: Save the best model
    print("\n7. Saving the best model...")
    try:
        ml_pipeline.save_model(ml_pipeline.best_model_name)
        print(f"✅ Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Step 8: Make predictions on new data
    print("\n8. Making predictions on new data...")
    if len(ehr_data) > len(X_test):
        # Use some data for prediction demonstration
        new_data_embeddings = np.stack(ehr_data['embedding'].values[-3:])
        predictions = ml_pipeline.predict(new_data_embeddings)
        
        print("Sample predictions:")
        for i, pred in enumerate(predictions):
            print(f"  Patient {i+1}: {'High Risk' if pred == 1 else 'Low Risk'} (prediction: {pred})")
    
    print("\n" + "=" * 60)
    print("✅ Example completed successfully!")
    
    if not use_real_api:
        print("\n💡 To use real Gemini embeddings:")
        print("   1. Get a Google API key from https://makersuite.google.com/app/apikey")
        print("   2. Set it as GOOGLE_API_KEY environment variable")
        print("   3. Run this script again")


def demonstrate_advanced_features():
    """
    Demonstrate advanced features of the pipeline
    """
    print("\n" + "=" * 60)
    print("Advanced Features Demonstration")
    print("=" * 60)
    
    # Create larger sample dataset
    data = create_sample_ehr_data(n_samples=500)
    
    # Initialize ML pipeline
    ml_pipeline = EHRMLPipeline(task_type="classification")
    
    # Prepare data
    X_train, X_test, y_train, y_test = ml_pipeline.prepare_data(
        df=data, 
        target_column='readmission_risk'
    )
    
    # Hyperparameter tuning for Random Forest
    print("\n1. Hyperparameter tuning for Random Forest...")
    tuning_results = ml_pipeline.hyperparameter_tuning(
        X_train, y_train, 
        model_name="random_forest"
    )
    print(f"Best parameters: {tuning_results['best_params']}")
    print(f"Best CV score: {tuning_results['best_score']:.4f}")
    
    # Compare with default model
    print("\n2. Comparing tuned vs default model performance...")
    tuned_metrics = ml_pipeline.evaluate_model(
        tuning_results['best_model'], X_test, y_test
    )
    
    # Train default model for comparison
    default_pipeline = EHRMLPipeline(task_type="classification")
    default_pipeline.prepare_data(df=data, target_column='readmission_risk')
    default_results = default_pipeline.train_models(X_train, y_train)
    default_metrics = default_pipeline.evaluate_model(
        default_pipeline.models['random_forest'], X_test, y_test
    )
    
    print("Performance Comparison:")
    print(f"  Tuned RF Accuracy:   {tuned_metrics['accuracy']:.4f}")
    print(f"  Default RF Accuracy: {default_metrics['accuracy']:.4f}")
    
    print("\n✅ Advanced features demonstration completed!")


if __name__ == "__main__":
    # Run basic example
    main()
    
    # Run advanced features demonstration
    demonstrate_advanced_features() 