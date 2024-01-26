from django.urls import path

from lolpredictor import views

urlpatterns = [
    path("api/status/", views.status, name="status"),
    path("api/reload/", views.reload, name="reload"),
    path("api/players/", views.players, name="players"),
    path("api/players/<str:query>/", views.player, name="player"),
    path("api/partners/<str:query>/", views.partners, name="partners"),
    path("api/pair/", views.pair, name="pair"),
    path("api/team/", views.team, name="team"),
]
