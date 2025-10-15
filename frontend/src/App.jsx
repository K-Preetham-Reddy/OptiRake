import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import Dashboard from './pages/Dashboard';
import Plans from './pages/Plans';
import Optimization from './pages/Optimization';
import Results from './pages/Results';
import { Toaster } from 'react-hot-toast';

function App() {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate loading time to ensure CSS is loaded
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 100);

    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading Rake Formation System...</p>
        </div>
      </div>
    );
  }

  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <main className="pt-16">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/plans" element={<Plans />} />
            <Route path="/optimize" element={<Optimization />} />
            <Route path="/results/:jobId" element={<Results />} />
            <Route path="*" element={<Dashboard />} />
          </Routes>
        </main>
        <Toaster position="top-right" />
      </div>
    </Router>
  );
}

export default App;