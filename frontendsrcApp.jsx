import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';

// Componentes
import AlertList from './components/AlertList';
import DetectionViewer from './components/DetectionViewer';
import StatsCard from './components/StatsCard';
import MapView from './components/MapView';
import ImageUploader from './components/ImageUploader