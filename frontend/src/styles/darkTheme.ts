// Dark theme styles for PropPulse
export const darkThemeStyles = {
  // Layout styles
  pageContainer: `
    min-h-screen 
    bg-[#0A0A0B] 
    text-white 
    p-6
  `,
  
  // Card styles
  card: `
    bg-[#18181B] 
    rounded-lg 
    shadow-xl 
    border 
    border-[#27272A]
    overflow-hidden
    backdrop-filter 
    backdrop-blur-lg
  `,

  // Card header styles
  cardHeader: `
    p-4 
    border-b 
    border-[#27272A]
  `,

  // Text styles
  heading: `
    text-2xl 
    font-bold 
    bg-gradient-to-r from-blue-400 to-indigo-400
    text-transparent 
    bg-clip-text
  `,
  
  subheading: `
    text-lg 
    text-gray-200
    font-medium
  `,

  // Stats card styles
  statsCard: `
    bg-[#18181B] 
    rounded-lg 
    p-6 
    flex 
    flex-col 
    justify-between
    border 
    border-[#27272A]
    hover:border-[#3F3F46]
    transition-colors
    duration-200
  `,

  // Value text styles
  valueText: `
    text-3xl 
    font-bold 
    bg-gradient-to-r from-blue-400 to-indigo-400
    text-transparent 
    bg-clip-text
  `,

  labelText: `
    text-sm 
    text-gray-400
    font-medium
  `,

  // Button styles
  button: `
    bg-blue-500 
    hover:bg-blue-600 
    text-white 
    font-semibold 
    py-2 
    px-4 
    rounded-lg
    transition-all
    duration-200
    hover:shadow-lg
    hover:shadow-blue-500/20
  `,

  // Tab styles
  tab: `
    px-4 
    py-2 
    rounded-lg 
    font-medium 
    transition-all
    duration-200
  `,
  
  activeTab: `
    bg-blue-500 
    text-white
    shadow-lg
    shadow-blue-500/20
  `,

  inactiveTab: `
    text-gray-400 
    hover:text-gray-200 
    hover:bg-[#27272A]
  `,

  // New gradient background style
  gradientBg: `
    bg-gradient-to-r from-[#0ea5e9] to-[#1d4ed8]
  `,

  // Chart container styles
  chartContainer: `
    bg-[#18181B] 
    rounded-xl 
    p-6 
    border 
    border-[#27272A]
    backdrop-filter 
    backdrop-blur-lg
  `,

  // Grid styles
  grid: `
    grid 
    grid-cols-1 
    md:grid-cols-2 
    lg:grid-cols-3 
    gap-6
  `,

  // Table styles
  table: `
    w-full 
    border-collapse
  `,

  tableHeader: `
    bg-[#27272A] 
    text-left 
    text-sm 
    font-medium 
    text-gray-200 
    uppercase 
    tracking-wider
  `,

  tableCell: `
    border-b 
    border-[#27272A] 
    text-gray-300
  `,

  // Search input styles
  searchInput: `
    bg-[#27272A] 
    text-white 
    rounded-lg 
    px-4 
    py-2 
    w-full 
    focus:outline-none 
    focus:ring-2 
    focus:ring-blue-500
    border
    border-[#3F3F46]
    hover:border-[#52525B]
    transition-colors
    duration-200
  `,

  // Badge styles
  badge: {
    success: `
      bg-emerald-900/50 
      text-emerald-300 
      px-3 
      py-1 
      rounded-full 
      text-sm
      border
      border-emerald-800/50
    `,
    warning: `
      bg-amber-900/50 
      text-amber-300 
      px-3 
      py-1 
      rounded-full 
      text-sm
      border
      border-amber-800/50
    `,
    error: `
      bg-rose-900/50 
      text-rose-300 
      px-3 
      py-1 
      rounded-full 
      text-sm
      border
      border-rose-800/50
    `
  }
}; 