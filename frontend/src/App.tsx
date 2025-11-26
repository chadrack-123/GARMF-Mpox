import { Routes, Route } from 'react-router-dom'
import { Box } from '@mui/material'
import Layout from './components/Layout'
import StudiesDashboard from './pages/StudiesDashboard'
import StudyDetail from './pages/StudyDetail'
import DatasetManager from './pages/DatasetManager'
import ConfigBuilder from './pages/ConfigBuilder'
import RunMonitor from './pages/RunMonitor'
import XAIExplorer from './pages/XAIExplorer'
import DocumentationViewer from './pages/DocumentationViewer'
import ReleaseBundles from './pages/ReleaseBundles'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<StudiesDashboard />} />
        <Route path="/studies/:studyId" element={<StudyDetail />} />
        <Route path="/datasets" element={<DatasetManager />} />
        <Route path="/configs/:studyId" element={<ConfigBuilder />} />
        <Route path="/runs" element={<RunMonitor />} />
        <Route path="/runs/:runId/xai" element={<XAIExplorer />} />
        <Route path="/runs/:runId/docs" element={<DocumentationViewer />} />
        <Route path="/bundles" element={<ReleaseBundles />} />
      </Routes>
    </Layout>
  )
}

export default App
