import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd()))

def main():
    print("Verifying imports...")
    
    try:
        from src.config import Settings
        print("Config imported")
        
        print("Orchestrator imported")
        
        from src.pipeline.video import VideoEngine
        print("VideoEngine imported")
        
        from src.pipeline.upscale import UpscaleEngine
        print("UpscaleEngine imported")
        
        from src.integrations.whatsapp import WhatsAppClient
        print("WhatsAppClient imported")
        
        print("Routes imported")
        
        print("Main app imported")
        
        # Test basic instantiation
        config = Settings()
        print("Settings instantiated")
        
        video = VideoEngine(config)
        print("VideoEngine instantiated")
        
        upscale = UpscaleEngine(config)
        print("UpscaleEngine instantiated")
        
        whatsapp = WhatsAppClient(config)
        print("WhatsAppClient instantiated")
        
        print("Verification successful!")
        
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
