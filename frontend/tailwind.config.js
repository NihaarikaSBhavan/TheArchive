/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        'heading': ['Playfair Display', 'serif'],
        'body': ['Inter', 'sans-serif'],
        'mono': ['JetBrains Mono', 'monospace'],
      },
      borderRadius: {
        lg: '0px',
        md: '0px',
        sm: '0px',
        DEFAULT: '0px',
      },
      colors: {
        background: '#F9F9F7',
        foreground: '#1A1A1A',
        card: {
          DEFAULT: '#FFFFFF',
          foreground: '#1A1A1A',
        },
        popover: {
          DEFAULT: '#FFFFFF',
          foreground: '#1A1A1A',
        },
        primary: {
          DEFAULT: '#CC5500',
          foreground: '#FFFFFF',
        },
        secondary: {
          DEFAULT: '#5A5A55',
          foreground: '#FFFFFF',
        },
        muted: {
          DEFAULT: '#F0F0ED',
          foreground: '#5A5A55',
        },
        accent: {
          DEFAULT: '#E67E22',
          foreground: '#FFFFFF',
        },
        destructive: {
          DEFAULT: '#D32F2F',
          foreground: '#FFFFFF',
        },
        border: '#E5E5E0',
        input: '#E5E5E0',
        ring: '#CC5500',
        sidebar: '#F0F0ED',
      },
      boxShadow: {
        'brutalist': '4px 4px 0px 0px #1A1A1A',
        'brutalist-lg': '8px 8px 0px 0px #1A1A1A',
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
