import React, { useEffect, useState } from 'react';

const cityImages = [
  'https://images.unsplash.com/photo-1449824913935-59a10b8d2000', // NYC
  'https://images.unsplash.com/photo-1444723121867-7a241cacace9', // Chicago
  'https://images.unsplash.com/photo-1502010183644-c94527b83c7e', // Seattle
  'https://images.unsplash.com/photo-1515963835374-c58c5444e4c6', // LA
  'https://images.unsplash.com/photo-1470219556762-1771e7f9427d', // Boston
];

const CityBackground: React.FC = () => {
  const [currentImageIndex] = useState(() => 
    Math.floor(Math.random() * cityImages.length)
  );

  return (
    <div className="absolute inset-0 z-0 overflow-hidden">
      <div 
        className="absolute inset-0 bg-cover bg-center transition-opacity duration-1000"
        style={{
          backgroundImage: `url(${cityImages[currentImageIndex]})`,
        }}
      />
      <div className="absolute inset-0 bg-gradient-to-b from-[#0A0A0B]/70 via-[#0A0A0B]/80 to-[#0A0A0B]/95 backdrop-blur-[2px]" />
    </div>
  );
};

export default CityBackground; 