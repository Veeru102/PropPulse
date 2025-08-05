import React from 'react';
import { motion } from 'framer-motion';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

interface InvestmentScoreDialProps {
  score: number;
}

const InvestmentScoreDial: React.FC<InvestmentScoreDialProps> = ({ score }) => {
  // Color based on score
  const getColor = (value: number) => {
    if (value >= 70) return '#22c55e'; // green-500
    if (value >= 40) return '#eab308'; // yellow-500
    return '#ef4444'; // red-500
  };

  // Performance tier label
  const getTierLabel = (value: number) => {
    if (value >= 70) return 'Strong Investment';
    if (value >= 40) return 'Moderate Potential';
    return 'High Risk';
  };

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      whileInView={{ scale: 1, opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="flex flex-col items-center"
    >
      <div className="w-48 h-48">
        <CircularProgressbar
          value={score}
          text={`${score}`}
          circleRatio={0.75}
          styles={buildStyles({
            rotation: 1 / 2 + 1 / 8,
            strokeLinecap: 'round',
            trailColor: '#1e293b', // slate-800
            pathColor: getColor(score),
            textColor: 'white',
            textSize: '24px',
          })}
        />
      </div>
      <div className="mt-4 text-center">
        <h4 className="text-lg font-medium text-white">Investment Score</h4>
        <p className="text-sm text-slate-400">{getTierLabel(score)}</p>
      </div>
    </motion.div>
  );
};

export default InvestmentScoreDial; 