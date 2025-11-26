import { useState } from 'react'
import { useParams } from 'react-router-dom'
import {
  Container,
  Typography,
  TextField,
  Button,
  Box,
  Paper,
} from '@mui/material'

export default function ConfigBuilder() {
  const { studyId } = useParams<{ studyId: string }>()
  const [yamlText, setYamlText] = useState('')

  return (
    <Container maxWidth="xl">
      <Typography variant="h4" gutterBottom>
        Configuration Builder
      </Typography>
      
      <Box sx={{ display: 'flex', gap: 2, height: '70vh' }}>
        <Paper sx={{ flex: 1, p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Form Editor
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Configure your ML pipeline using the form
          </Typography>
        </Paper>

        <Paper sx={{ flex: 1, p: 2, display: 'flex', flexDirection: 'column' }}>
          <Typography variant="h6" gutterBottom>
            YAML Preview
          </Typography>
          <TextField
            multiline
            fullWidth
            value={yamlText}
            onChange={(e) => setYamlText(e.target.value)}
            placeholder="# YAML configuration will appear here"
            sx={{ flex: 1, fontFamily: 'monospace' }}
          />
          <Button variant="contained" sx={{ mt: 2 }}>
            Save Configuration
          </Button>
        </Paper>
      </Box>
    </Container>
  )
}
