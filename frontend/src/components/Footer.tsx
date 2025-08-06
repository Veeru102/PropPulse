import React from 'react';

const technologies = [
  { name: 'React', link: 'https://react.dev/' },
  { name: 'TypeScript', link: 'https://www.typescriptlang.org/' },
  { name: 'Tailwind CSS', link: 'https://tailwindcss.com/' },
  { name: 'FastAPI', link: 'https://fastapi.tiangolo.com/' },
  { name: 'OpenAI GPT-4', link: 'https://openai.com/gpt-4/' },
  { name: 'Mapbox', link: 'https://www.mapbox.com/' },
  { name: 'Framer Motion', link: 'https://www.framer.com/motion/' },
  { name: 'PostgreSQL', link: 'https://www.postgresql.org/' },
  { name: 'React Query', link: 'https://tanstack.com/query/latest' },
  { name: 'PyTorch', link: 'https://pytorch.org/' },
  { name: 'Faiss', link: 'https://faiss.ai/' },
  { name: 'Scikit-learn', link: 'https://scikit-learn.org/stable/' },
  { name: 'Chart.js', link: 'https://www.chartjs.org/' },
];

const Footer: React.FC = () => {
  return (
    <footer className="bg-[#121212] text-slate-400 py-8 mt-16 border-t border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-6">
          <div className="flex flex-wrap gap-x-4 gap-y-2 justify-center">
            {technologies.map(({ name, link }) => (
              <a
                key={name}
                href={link}
                target="_blank"
                rel="noopener noreferrer"
                className="px-3 py-1 rounded-md bg-slate-800 text-sm font-medium text-slate-300 border border-slate-700 hover:bg-slate-700 hover:text-white transition-colors"
              >
                {name}
              </a>
            ))}
          </div>
          <div className="flex items-center gap-3">
            <a
              href="https://github.com/Veeru102"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-400 hover:text-white transition-colors"
              aria-label="GitHub Profile"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="w-5 h-5"
              >
                <path
                  d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.091.682-.217.682-.483 0-.237-.009-.868-.014-1.703-2.782.604-3.369-1.342-3.369-1.342-.455-1.158-1.11-1.467-1.11-1.467-.908-.62.069-.608.069-.608 1.004.07 1.532 1.032 1.532 1.032.892 1.53 2.341 1.087 2.91.832.091-.647.35-1.087.636-1.337-2.221-.253-4.555-1.112-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.27.098-2.645 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.85.004 1.71.115 2.512.337 1.909-1.296 2.747-1.026 2.747-1.026.546 1.375.203 2.392.1 2.645.64.7 1.028 1.595 1.028 2.688 0 3.848-2.337 4.695-4.566 4.943.359.309.678.919.678 1.852 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.481A10.003 10.003 0 0 0 22 12c0-5.523-4.477-10-10-10z"
                />
              </svg>
            </a>
            <p className="text-sm">Built by Veeru Senthil</p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
