import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Box,
  Container,
  Tab,
  Tabs,
  Typography,
  Paper,
} from '@mui/material'
import { useState } from 'react'
import { getStudy, getStudyDatasets, getStudyConfigs } from '../api/client'

export default function StudyDetail() {
  const { studyId } = useParams<{ studyId: string }>()
  const [tab, setTab] = useState(0)

  const { data: study } = useQuery({
    queryKey: ['study', studyId],
    queryFn: async () => {
      const response = await getStudy(Number(studyId))
      return response.data
    },
  })

  const { data: datasets } = useQuery({
    queryKey: ['datasets', studyId],
    queryFn: async () => {
      const response = await getStudyDatasets(Number(studyId))
      return response.data
    },
  })

  const { data: configs } = useQuery({
    queryKey: ['configs', studyId],
    queryFn: async () => {
      const response = await getStudyConfigs(Number(studyId))
      return response.data
    },
  })

  return (
    <Container maxWidth="xl">
      <Typography variant="h4" gutterBottom>
        {study?.title}
      </Typography>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tab} onChange={(_, v) => setTab(v)}>
          <Tab label="Overview" />
          <Tab label="Datasets" />
          <Tab label="Configs" />
          <Tab label="Runs" />
        </Tabs>
      </Box>

      {tab === 0 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Study Information
          </Typography>
          <Typography><strong>Disease:</strong> {study?.disease}</Typography>
          <Typography><strong>Modality:</strong> {study?.modality}</Typography>
          {study?.doi && <Typography><strong>DOI:</strong> {study.doi}</Typography>}
          {study?.citation && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2">Citation:</Typography>
              <Typography variant="body2">{study.citation}</Typography>
            </Box>
          )}
        </Paper>
      )}

      {tab === 1 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Datasets ({datasets?.length || 0})
          </Typography>
          {datasets?.map((dataset: any) => (
            <Box key={dataset.id} sx={{ mb: 2, p: 2, border: '1px solid #ddd', borderRadius: 1 }}>
              <Typography><strong>Type:</strong> {dataset.type}</Typography>
              <Typography><strong>URI:</strong> {dataset.storage_uri}</Typography>
            </Box>
          ))}
        </Paper>
      )}

      {tab === 2 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Configurations ({configs?.length || 0})
          </Typography>
          {configs?.map((config: any) => (
            <Box key={config.id} sx={{ mb: 2, p: 2, border: '1px solid #ddd', borderRadius: 1 }}>
              <Typography><strong>Name:</strong> {config.name}</Typography>
              <Typography><strong>Type:</strong> {config.config_type}</Typography>
            </Box>
          ))}
        </Paper>
      )}
    </Container>
  )
}
