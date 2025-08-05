import React from 'react';
import { motion } from 'framer-motion';

const PropertySearchDemo: React.FC = () => {
  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      whileInView={{ scale: 1, opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="relative rounded-xl overflow-hidden shadow-2xl bg-slate-800"
    >
      <video
        src="/videos/map-search-demo.mp4"
        autoPlay
        loop
        muted
        playsInline
        className="w-full"
        onError={(e) => {
          console.error('Video loading error:', e);
          const target = e.target as HTMLVideoElement;
          target.style.display = 'none';
          target.parentElement?.classList.add('min-h-[300px]', 'flex', 'items-center', 'justify-center');
          const errorMsg = document.createElement('div');
          errorMsg.className = 'text-slate-500 text-center p-8';
          errorMsg.textContent = 'Interactive property search demo';
          target.parentElement?.appendChild(errorMsg);
        }}
      />
    </motion.div>
  );
};

export default PropertySearchDemo; 