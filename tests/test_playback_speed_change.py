#!/usr/bin/env python3
"""
Test to reproduce the playback speed change issue that causes a blank white screen.
"""

import asyncio
import aiohttp
import time
import json
from typing import Dict, Any, Optional

class PlaybackSpeedChangeTest:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.websocket_url = "ws://localhost:8000/ws"
        self.test_results = []
        
    async def test_playback_speed_change(self):
        """Test changing playback speed during active playback"""
        print("\n=== Testing Playback Speed Change Issue ===")
        
        try:
            # Step 1: Get available checkpoints
            checkpoints = await self.get_checkpoints()
            if not checkpoints:
                print("❌ No checkpoints available for testing")
                return False
                
            checkpoint_id = checkpoints[0]['id']
            print(f"✅ Using checkpoint: {checkpoint_id}")
            
            # Step 2: Start playback at normal speed
            print("Starting playback at 1.0x speed...")
            success = await self.start_playback(checkpoint_id, 1.0)
            if not success:
                print("❌ Failed to start playback")
                return False
                
            # Step 3: Wait for playback to begin and get some data
            print("Waiting for playback data...")
            await asyncio.sleep(3)
            
            # Step 4: Check if we're receiving data
            status = await self.get_playback_status()
            if not status or not status.get('is_playing'):
                print("❌ Playback not running")
                return False
                
            print("✅ Playback is running")
            
            # Step 5: Change speed to 2.0x
            print("Changing speed to 2.0x...")
            success = await self.set_playback_speed(2.0)
            if not success:
                print("❌ Failed to change playback speed")
                return False
                
            # Step 6: Wait and check if data continues
            print("Waiting after speed change...")
            await asyncio.sleep(5)
            
            # Step 7: Check status again
            status_after = await self.get_playback_status()
            if not status_after:
                print("❌ Could not get status after speed change")
                return False
                
            print(f"✅ Status after speed change: playing={status_after.get('is_playing')}, speed={status_after.get('playback_speed')}")
            
            # Step 8: Test multiple speed changes
            speeds_to_test = [1.5, 3.0, 0.5, 1.0]
            for speed in speeds_to_test:
                print(f"Testing speed change to {speed}x...")
                success = await self.set_playback_speed(speed)
                if not success:
                    print(f"❌ Failed to change speed to {speed}x")
                    return False
                    
                await asyncio.sleep(2)
                
                # Check if playback is still running
                status = await self.get_playback_status()
                if not status or not status.get('is_playing'):
                    print(f"❌ Playback stopped after speed change to {speed}x")
                    return False
                    
                print(f"✅ Speed {speed}x working")
            
            # Step 9: Stop playback
            print("Stopping playback...")
            await self.stop_playback()
            
            print("✅ All speed change tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def get_checkpoints(self) -> list:
        """Get available checkpoints"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/checkpoints") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"Failed to get checkpoints: {response.status}")
                        return []
        except Exception as e:
            print(f"Error getting checkpoints: {e}")
            return []
    
    async def start_playback(self, checkpoint_id: str, speed: float) -> bool:
        """Start playback for a checkpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/checkpoints/{checkpoint_id}/playback/start",
                    json={"speed": speed}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ Playback started: {result.get('message')}")
                        return True
                    else:
                        print(f"❌ Failed to start playback: {response.status}")
                        return False
        except Exception as e:
            print(f"Error starting playback: {e}")
            return False
    
    async def set_playback_speed(self, speed: float) -> bool:
        """Set playback speed"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/checkpoints/playback/speed",
                    json={"speed": speed}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ Speed set to {speed}x: {result.get('message')}")
                        return True
                    else:
                        print(f"❌ Failed to set speed: {response.status}")
                        return False
        except Exception as e:
            print(f"Error setting speed: {e}")
            return False
    
    async def get_playback_status(self) -> Optional[Dict[str, Any]]:
        """Get current playback status"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/checkpoints/playback/status") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"Failed to get status: {response.status}")
                        return None
        except Exception as e:
            print(f"Error getting status: {e}")
            return None
    
    async def stop_playback(self) -> bool:
        """Stop playback"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/checkpoints/playback/stop") as response:
                    if response.status == 200:
                        print("✅ Playback stopped")
                        return True
                    else:
                        print(f"❌ Failed to stop playback: {response.status}")
                        return False
        except Exception as e:
            print(f"Error stopping playback: {e}")
            return False

async def main():
    """Run the playback speed change test"""
    test = PlaybackSpeedChangeTest()
    
    print("Starting playback speed change test...")
    print("This test will:")
    print("1. Start playback at 1.0x speed")
    print("2. Change speed to 2.0x")
    print("3. Test multiple speed changes")
    print("4. Verify playback continues working")
    print()
    
    success = await test.test_playback_speed_change()
    
    if success:
        print("\n🎉 Playback speed change test PASSED")
        print("The issue may be frontend-specific or related to websocket handling")
    else:
        print("\n💥 Playback speed change test FAILED")
        print("The issue is likely in the backend playback system")
    
    return success

if __name__ == "__main__":
    asyncio.run(main()) 