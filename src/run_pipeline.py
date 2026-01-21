"""
Complete Pipeline Executor
Runs the entire medical AI system pipeline from data preprocessing to deployment
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineExecutor:
    """
    Manages the complete ML pipeline execution
    """
    
    def __init__(self):
        self.steps_completed = []
        self.total_steps = 6
    
    def print_header(self, step_num, step_name):
        """Print formatted step header"""
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP {step_num}/{self.total_steps}: {step_name}")
        logger.info("=" * 80 + "\n")
    
    def run_step(self, step_num, step_name, module_name):
        """Execute a pipeline step"""
        self.print_header(step_num, step_name)
        
        try:
            # Import and run module
            module = __import__(module_name.replace('.py', ''))
            module.main()
            
            self.steps_completed.append(step_name)
            logger.info(f"✅ {step_name} completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in {step_name}: {str(e)}")
            logger.exception("Full traceback:")
            return False
    
    def run_full_pipeline(self):
        """Execute complete pipeline"""
        logger.info("\n" + "=" * 80)
        logger.info("SECURE MULTIMODAL MEDICAL DATA FUSION - COMPLETE PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80 + "\n")
        
        # Pipeline steps
        steps = [
            (1, "Data Preprocessing", "02_data_preprocessing"),
            (2, "Feature Extraction", "03_feature_extraction"),
            (3, "Model Training", "04_model_training"),
            (4, "Quantum Security Setup", "05_quantum_security"),
            (5, "Model Evaluation", "06_evaluation")
        ]
        
        # Execute steps
        for step_num, step_name, module_name in steps:
            success = self.run_step(step_num, step_name, module_name)
            
            if not success:
                logger.error(f"\n⚠️ Pipeline stopped at step {step_num} due to errors.")
                logger.info(f"Steps completed: {', '.join(self.steps_completed)}")
                return False
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Steps: {self.total_steps}")
        logger.info(f"Completed: {len(self.steps_completed)}")
        logger.info("Steps executed:")
        for i, step in enumerate(self.steps_completed, 1):
            logger.info(f"  {i}. {step} ✅")
        logger.info(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        return True
    
    def quick_setup(self):
        """Quick setup without full training (for demo)"""
        logger.info("\n" + "=" * 80)
        logger.info("QUICK SETUP MODE (Demo)")
        logger.info("=" * 80 + "\n")
        
        logger.info("This mode will:")
        logger.info("  1. Check configuration")
        logger.info("  2. Setup directories")
        logger.info("  3. Prepare for manual data upload")
        logger.info("  4. Ready web application")
        
        # Check config
        try:
            import project_config
            logger.info("✅ Configuration loaded")
        except:
            logger.error("❌ Configuration error")
            return False
        
        # Create directories
        from project_config import PROCESSED_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR
        for dir_path in [PROCESSED_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        logger.info("✅ Directories created")
        
        logger.info("\n📝 Next Steps:")
        logger.info("  1. Place your datasets in the D:\\medical_datasets folder")
        logger.info("  2. Run: python run_pipeline.py --full")
        logger.info("  3. Or run: streamlit run app.py (for demo with untrained models)")
        
        return True


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Medical AI Pipeline')
    parser.add_argument('--full', action='store_true', help='Run full pipeline including training')
    parser.add_argument('--quick', action='store_true', help='Quick setup without training')
    parser.add_argument('--app', action='store_true', help='Launch web application only')
    parser.add_argument('--eval', action='store_true', help='Run evaluation only')
    
    args = parser.parse_args()
    
    executor = PipelineExecutor()
    
    if args.app:
        logger.info("Launching web application...")
        os.system("streamlit run app.py")
    
    elif args.eval:
        logger.info("Running evaluation...")
        executor.run_step(5, "Model Evaluation", "06_evaluation")
    
    elif args.quick:
        executor.quick_setup()
    
    elif args.full:
        success = executor.run_full_pipeline()
        
        if success:
            logger.info("\n🎉 Pipeline completed successfully!")
            logger.info("\n📱 To launch the web application, run:")
            logger.info("   streamlit run app.py")
        else:
            logger.error("\n❌ Pipeline failed. Check logs for details.")
    
    else:
        logger.info("Medical AI Pipeline Executor")
        logger.info("\nUsage:")
        logger.info("  python run_pipeline.py --full    # Run complete pipeline")
        logger.info("  python run_pipeline.py --quick   # Quick setup only")
        logger.info("  python run_pipeline.py --app     # Launch web app")
        logger.info("  python run_pipeline.py --eval    # Run evaluation only")


if __name__ == "__main__":
    main()