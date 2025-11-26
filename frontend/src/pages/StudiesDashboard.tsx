import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Box,
  Button,
  Card,
  CardContent,
  CardActions,
  Container,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  TextField,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
} from '@mui/material'
import { Add as AddIcon } from '@mui/icons-material'
import { getStudies, createStudy } from '../api/client'
import type { Study } from '../types'

export default function StudiesDashboard() {
  const navigate = useNavigate()
  const [openDialog, setOpenDialog] = useState(false)
  const [formData, setFormData] = useState({
    title: '',
    citation: '',
    doi: '',
    disease: 'Mpox',
    modality: 'tabular' as 'image' | 'tabular' | 'mixed',
    notes: '',
  })

  const { data: studies, refetch } = useQuery({
    queryKey: ['studies'],
    queryFn: async () => {
      const response = await getStudies()
      return response.data as Study[]
    },
  })

  const handleSubmit = async () => {
    try {
      await createStudy(formData)
      setOpenDialog(false)
      refetch()
      setFormData({
        title: '',
        citation: '',
        doi: '',
        disease: 'Mpox',
        modality: 'tabular',
        notes: '',
      })
    } catch (error) {
      console.error('Failed to create study:', error)
    }
  }

  return (
    <Container maxWidth="xl">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Studies Dashboard
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setOpenDialog(true)}
        >
          New Study
        </Button>
      </Box>

      <Grid container spacing={3}>
        {studies?.map((study) => (
          <Grid item xs={12} sm={6} md={4} key={study.id}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {study.title}
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Disease: {study.disease}
                </Typography>
                <Chip
                  label={study.modality}
                  size="small"
                  color="primary"
                  sx={{ mt: 1 }}
                />
                {study.doi && (
                  <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                    DOI: {study.doi}
                  </Typography>
                )}
              </CardContent>
              <CardActions>
                <Button
                  size="small"
                  onClick={() => navigate(`/studies/${study.id}`)}
                >
                  View Details
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Study</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Title"
            fullWidth
            value={formData.title}
            onChange={(e) => setFormData({ ...formData, title: e.target.value })}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Citation"
            fullWidth
            multiline
            rows={2}
            value={formData.citation}
            onChange={(e) => setFormData({ ...formData, citation: e.target.value })}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="DOI"
            fullWidth
            value={formData.doi}
            onChange={(e) => setFormData({ ...formData, doi: e.target.value })}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Disease"
            fullWidth
            value={formData.disease}
            onChange={(e) => setFormData({ ...formData, disease: e.target.value })}
            sx={{ mb: 2 }}
          />
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Modality</InputLabel>
            <Select
              value={formData.modality}
              label="Modality"
              onChange={(e) =>
                setFormData({
                  ...formData,
                  modality: e.target.value as 'image' | 'tabular' | 'mixed',
                })
              }
            >
              <MenuItem value="tabular">Tabular</MenuItem>
              <MenuItem value="image">Image</MenuItem>
              <MenuItem value="mixed">Mixed</MenuItem>
            </Select>
          </FormControl>
          <TextField
            margin="dense"
            label="Notes"
            fullWidth
            multiline
            rows={3}
            value={formData.notes}
            onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button onClick={handleSubmit} variant="contained">
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  )
}
