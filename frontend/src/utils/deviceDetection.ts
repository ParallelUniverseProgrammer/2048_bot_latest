import React from 'react';

export interface DeviceInfo {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  orientation: 'portrait' | 'landscape';
  screenWidth: number;
  screenHeight: number;
}

export type DisplayMode = 'mobile' | 'desktop';

export function getDeviceInfo(): DeviceInfo {
  const screenWidth = window.innerWidth
  const screenHeight = window.innerHeight
  const orientation: 'portrait' | 'landscape' = screenWidth > screenHeight ? 'landscape' : 'portrait'

  // Prefer UA Client-Hints (Chromium-based browsers) if available
  // navigator.userAgentData.mobile returns a boolean (true on mobile, false on desktop). Safari/iOS currently
  // does not implement UA-CH, so we fallback to classic UA string parsing for those cases.

  const uaData = (navigator as any).userAgentData
  const isMobileCH = typeof uaData?.mobile === 'boolean' ? uaData.mobile : null

  const ua = navigator.userAgent || navigator.vendor || (window as any).opera
  const uaLower = ua.toLowerCase()

  // Wide set of mobile identifiers (phones + tablets)
  const mobileRegex = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini|windows phone/i

  const isMobileUA = mobileRegex.test(uaLower)

  // Further refine tablet detection – iPad and some Android tablets report themselves as mobile
  const tabletRegex = /ipad|tablet|nexus 7|nexus 10|sm-t|kfapwi|silk/i
  const isTabletUA = tabletRegex.test(uaLower)

  // Decide device category, prioritising UA Client-Hints, then UA regex, then touch capability
  let isMobile = false
  let isTablet = false
  let isDesktop = false

  if (isMobileCH !== null) {
    // If UA-CH is available, use it.
    isMobile = isMobileCH
    isDesktop = !isMobileCH
  } else if (isTabletUA) {
    isTablet = true
  } else if (isMobileUA) {
    isMobile = true
  } else {
    // No mobile indicators, treat as desktop (touch-screen laptops will fall here)
    isDesktop = true
  }

  // Edge-case: some large tablets in landscape should be considered tablets even if UA-CH says mobile
  if (isMobile && screenWidth >= 1024) {
    isMobile = false
    isTablet = true
  }
   
  return {
    isMobile,
    isTablet,
    isDesktop,
    orientation,
    screenWidth,
    screenHeight
  };
}

export function getDisplayMode(): DisplayMode {
  const device = getDeviceInfo();
  
  // Mobile devices always use mobile mode
  if (device.isMobile) {
    return 'mobile';
  }
  
  // Tablets use mobile mode in portrait, desktop mode in landscape
  if (device.isTablet) {
    return device.orientation === 'portrait' ? 'mobile' : 'desktop';
  }
  
  // Desktop devices always use desktop mode
  return 'desktop';
}

export function useDeviceDetection() {
  const [deviceInfo, setDeviceInfo] = React.useState<DeviceInfo>(getDeviceInfo());
  const [displayMode, setDisplayMode] = React.useState<DisplayMode>(getDisplayMode());
  
  React.useEffect(() => {
    const handleResize = () => {
      const newDeviceInfo = getDeviceInfo();
      const newDisplayMode = getDisplayMode();
      
      setDeviceInfo(newDeviceInfo);
      setDisplayMode(newDisplayMode);
    };
    
    // Listen for both resize and orientation change
    window.addEventListener('resize', handleResize);
    window.addEventListener('orientationchange', handleResize);
    
    // Also listen for orientation change event (mobile specific)
    if ('screen' in window && 'orientation' in window.screen) {
      window.screen.orientation?.addEventListener('change', handleResize);
    }
    
    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleResize);
      if ('screen' in window && 'orientation' in window.screen) {
        window.screen.orientation?.removeEventListener('change', handleResize);
      }
    };
  }, []);
  
  return { deviceInfo, displayMode };
} 