import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const terms = [
  'AI-Powered',
  'Machine-Learning Driven',
  'Real-Time',
  'Investor-Focused',
  'Automated',
  'Decision-Ready',
  'Deal-Oriented'
];

const AnimatedHeading: React.FC = () => {
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % terms.length);
    }, 2500);

    return () => clearInterval(interval);
  }, []);

  return (
    <h1 className="text-5xl font-bold tracking-tight text-white">
      <AnimatePresence mode="wait">
        <motion.span
          key={currentIndex}
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: -20, opacity: 0 }}
          transition={{ duration: 0.5 }}
          className="inline-block"
        >
          {terms[currentIndex]}
        </motion.span>
      </AnimatePresence>
      <br />
      <span className="text-blue-400">Real Estate Analysis</span>
    </h1>
  );
};

export default AnimatedHeading; 